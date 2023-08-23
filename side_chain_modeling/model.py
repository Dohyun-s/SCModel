import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.utils.checkpoint
import torch.nn.functional as F
import math
from torch import einsum

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
        self.scpred = SCPred(d_hidden=hidden_dim) 
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        edge_in = num_positional_embeddings * 8 #+ self.num_rbf*25
        self.ln_post = nn.LayerNorm(hidden_dim)
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.attention_bias = AttentionWithBias(d_in=128, d_bias=32, n_head=8, d_hidden=16)
        #         self.structure_projection = nn.Parameter(torch.randn(128, 512))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
    
    def forward(self, dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=dist_ca.device
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, residue_idx, chain_encoding_all)
        # h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        # h_E = self.W_e(E)

        # # Encoder is unmasked self-attention
        # mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        # mask_attend = mask.unsqueeze(-1) * mask_attend
        # for layer in self.encoder_layers:
        #     h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
        h_V = V.to(E.device)
        E_idx = E_idx.to(E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
#             h_V, h_E = torch.utils.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_EV = self.attention_bias(h_V, h_E, E_idx, mask_attend) + h_V
        result = self.scpred(h_EV)
        return result
    
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
        
        node_in = 7 # dihedral 6 + residue_idx 1
#         self.node_embedding = nn.Linear(node_in, edge_features, bias=False)
        self.node_embedding = nn.Embedding(22, 6, padding_idx=21)
        self.node_embedding2 = nn.Linear(12, edge_features, bias=True)
        self.norm_nodes = nn.LayerNorm(edge_features)

    def _dist(self, dist_ca, mask_angle, eps=1E-6):
        D = mask_angle * dist_ca
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_angle) * D_max
        # number of Ca atoms is 14.
        Ca_dim = 32
        D_neighbors, E_idx = torch.topk(D_adjust, np.maximum(self.top_k, Ca_dim), dim=-1, largest=False)
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
    
    def forward(self, dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, residue_idx, chain_encoding_all):
        D_neighbors, E_idx = self._dist(dist_ca, mask_angle)
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
        
#         V = node_embe.cat((torch.unsqueeze(S, -1), dihedral), dim=-1)
        V = self.node_embedding(S)
        # V = V + dihedral
        V = torch.cat((V, dihedral), dim=-1)
        V = self.node_embedding2(V)
        V = self.norm_nodes(V)
        return V, E, E_idx

    
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

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features



class SCPred(nn.Module):
    def __init__(self, d_hidden=128):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_hidden)
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.NTOTAL = 10
        self.state = 10
        self.linear_out = nn.Linear(d_hidden, 2*self.state)

        self.reset_parameter()

    def reset_parameter(self):
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq):
        '''
        Predict side-chain torsion angles along with backbone torsions
        '''
        B, L = seq.shape[:2]
        si = self.norm_s0(seq)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        si /= torch.sqrt(torch.sum(torch.square(si), axis=-1, keepdims=True) + 1e-8)
        return si.view(B, L, self.NTOTAL, 2)

def init_lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module


class AttentionWithBias(nn.Module):
    def __init__(self, d_in=128, d_bias=32, n_head=8, d_hidden=32):
        super(AttentionWithBias, self).__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_bias = nn.LayerNorm(d_in)
        self.norm_bias2 = nn.LayerNorm(n_head)
        
        self.query_key = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.value = nn.Linear(d_in, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_in, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_in)

        self.scaling = 1 / math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.query_key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the beginning)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make sure the residual operation is the same as the Identity at the beginning
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, bias, E_idx, mask_attend):
        B, L = x.shape[:2]
        device = x.device
        
        # Combine normalization and transformation
        x_norm = self.norm_in(x)
        x_query_key = self.query_key(x_norm).reshape(B, L, self.h, self.dim)
        x_value = self.value(x_norm).reshape(B, L, self.h, self.dim)
        
        # Combine bias normalization
        bias = self.norm_bias(bias)
#         bias = self.norm_bias2(bias)

        B, L, I, H = bias.shape
        input_tensor = torch.zeros((B, L, L, H), device=device)
        expanded_E_idx = E_idx.unsqueeze(3).expand(-1, -1, -1, H)
        bias = torch.scatter(input_tensor, 2, expanded_E_idx, bias)
        bias = self.to_b(bias)
        
        # Compute the gating mechanism
        gate = torch.sigmoid(self.to_g(x_norm))
        
        # Compute the attention weights using batch matrix multiplication (bmm)
        query_key_scaled = x_query_key * self.scaling
        attn = einsum('bqhd,bkhd->bqkh', query_key_scaled, x_query_key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value
        out = einsum('bqkh,bkhd->bqhd', attn, x_value).reshape(B, L, -1)
        
        # Apply the gating mechanism
        out = gate * out
        #
        out = self.to_out(out)
        return out
