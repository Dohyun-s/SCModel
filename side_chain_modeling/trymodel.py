import torch.nn as nn
import numpy as np
from einops import rearrange
class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=32,
        augment_eps=0.1,
        dropout=0.1,
        num_positional_embeddings=16,
        num_rbf=16
    ):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.top_k = k_neighbors
        self.features = ProteinFeatures(edge_features=edge_features, num_positional_embeddings=num_positional_embeddings,\
                                        k_neighbors=k_neighbors, num_rbf=num_rbf)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

#         # Encoder layers
#         self.encoder_layers = nn.ModuleList([
#             EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
#             for _ in range(num_encoder_layers)
#         ])
        edge_in = num_positional_embeddings * 8 #+ self.num_rbf*25
        self.ln_post = nn.LayerNorm(hidden_dim)
        self.embeddings = PositionalEncodings(num_positional_embeddings)

        # self.structure_projection = nn.Parameter(torch.randn(128, 512))

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
    
    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B
    
    def forward(self, dist_ca, omega, theta, phi, dihedral, mask, S, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=dist_ca.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(dist_ca, omega, theta, phi, dihedral, mask, residue_idx, chain_encoding_all)
        node_v = dihedral
        # h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        # h_E = self.W_e(E)

        # # Encoder is unmasked self-attention
        # mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        # mask_attend = mask.unsqueeze(-1) * mask_attend
        # for layer in self.encoder_layers:
        #     h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        return E, E_idx

    
class ProteinFeatures(nn.Module):
    def __init__(self, edge_features=128, num_positional_embeddings=16,
        k_neighbors=32, num_rbf=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        # Hyperparameters
        self.edge_features = edge_features
        self.top_k = k_neighbors

        edge_in = num_positional_embeddings * 8
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.num_rbf = num_rbf
        
    def _dist(self, dist_ca, mask, eps=1E-6):
        D = mask * dist_ca
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask) * D_max
        # number of Ca atoms is 14.
        Ca_dim = 14
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, Ca_dim), dim=-1, largest=False)
        return D_neighbors, E_idx
    
    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF
    
    def forward(self, dist_ca, omega, theta, phi, dihedral, mask, residue_idx, chain_encoding_all):
        D_neighbors, E_idx = self._dist(dist_ca, mask)
        offset = residue_idx[:,:,None] - residue_idx[:,None,:]
        
        edge_s = [offset, torch.cos(omega), torch.sin(omega), torch.cos(theta), \
                    torch.sin(theta), torch.cos(phi), torch.sin(phi)]
        edge_s = torch.cat([gather_edges(X[:,:,:,None], E_idx)[:,:,:,0] for X in edge_s])
        d_chains = ((chain_encoding_all[:, :, None] - chain_encoding_all[:,None,:])==0).long()
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_chains = torch.tile(E_chains,(7,1,1))
        E_positional = self.embeddings(edge_s.long(), E_chains)
        E_positional = rearrange(E_positional, '(n b) l t c -> b l t (n c)', n=7)
        
        RBF_all = self._rbf(D_neighbors)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)  # positional + ca-distance
        return E, E_idx

    
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features 