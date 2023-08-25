import torch
import scipy
import numpy as np
import torch.nn.functional as F
import random

def get_coords6d(xyz, dmax):
    xyz = xyz.permute(1,0,2)
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

    # recreate Cb given N,Ca,C
    Cb = generate_Cbeta(N,Ca,C)
    ### Exception if Cb is all Nan. Cannot calculate coords.
    if Cb.isnan().sum() == len(Cb) * 3:
        raise Exception("Condition not satisfied.")
    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    
    # Ca-Ca distance matrix
    dist_ca = np.full((nres, nres),999.9, dtype=np.float32)
    dist_ca[idx0,idx1] = np.linalg.norm(Ca[idx1]-Ca[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres), dtype=np.float32)
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres), dtype=np.float32)
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres), dtype=np.float32)
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist_ca, omega6d, theta6d, phi6d

def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    # From GVP-Pytorch
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                           res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                           res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                           res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                           res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            all_atoms = torch.Tensor(t['xyz'][res,])[0,] #[L, 14, 3]
                            try:
                                dist_ca, omega6d, theta6d, phi6d = get_coords6d(all_atoms, 20.0)
                            except Exception:
                                continue
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            coords_dict_chain['dihedral_'+letter]=_dihedrals(all_atoms)
                            coords_dict_chain['omega'+letter]=omega6d
                            coords_dict_chain['theta'+letter]=theta6d
                            coords_dict_chain['phi'+letter]=phi6d
                            coords_dict_chain['dist_ca_'+letter]=dist_ca
                            coords_dict_chain['xyz_coords']=all_atoms
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            # coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            # coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            # coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            # coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list

def generate_Cbeta(N,Ca,C):
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    #Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb

# calculate dihedral angles defined by 4 sets of points
def get_dihedrals(a, b, c, d):
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - torch.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - torch.sum(b2*b1, axis=-1)[:,None]*b1

    x = torch.sum(v*w, axis=-1)
    y = torch.sum(torch.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = torch.sum(v*w, axis=1)

    #return np.arccos(x)
    return np.arccos(np.clip(x, -1.0, 1.0))

def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    L_max = max([len(b['seq']) for b in batch])
    dist_ca, omega, theta, phi, mask_angle = [np.zeros([B, L_max, L_max]) for _ in range(5)]
    dihedral = np.zeros([B, L_max, 6])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
#     mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    X = np.zeros([B, L_max, 4, 3])
    
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        
        dist_ca_list, omega_list, phi_list, theta_list, dihedral_list = [], [], [], [], []
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        mask_list = []
        c = 1
        l0 = 0
        l1 = 0
        chain_len_list = [len(b[f'seq_chain_{letter}']) for letter in all_chains]
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                
                dist_ca_list.append(chain_coords[f'dist_ca_{letter}'])
                omega_list.append(chain_coords[f'omega{letter}'])
                theta_list.append(chain_coords[f'theta{letter}'])
                phi_list.append(chain_coords[f'phi{letter}'])
                dihedral_list.append(chain_coords[f'dihedral_{letter}'])
                mask_list.append(np.ones(chain_coords[f'dist_ca_{letter}'].shape))
                
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
#                 mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                
                dist_ca_list.append(chain_coords[f'dist_ca_{letter}'])
                omega_list.append(chain_coords[f'omega{letter}'])
                theta_list.append(chain_coords[f'theta{letter}'])
                phi_list.append(chain_coords[f'phi{letter}'])
#                 dihedral_temp = np.pad(chain_coords[f'dihedral_{letter}'], [[0,L_max-nres],[0,0]])
                dihedral_list.append(chain_coords[f'dihedral_{letter}'])
                mask_list.append(np.ones(chain_coords[f'dist_ca_{letter}'].shape))
        
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
#                 mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        dist_ca[i] = make_block_diagonal(dist_ca_list, L_max)
        omega[i] = make_block_diagonal(omega_list, L_max)
        theta[i] = make_block_diagonal(theta_list, L_max)
        phi[i] = make_block_diagonal(phi_list, L_max)
        dihedral_temp = np.concatenate(dihedral_list, 0)
        nres = dihedral_temp.shape[0]
        dihedral[i] = np.pad(dihedral_temp, [[0,L_max-nres],[0,0]])
        mask_angle[i] = make_block_diagonal(mask_list, L_max)
        
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        S[i, l:] = 21

    dist_ca[np.isnan(dist_ca)] = 0.
    omega[np.isnan(omega)] = 0.
    theta[np.isnan(theta)] = 0.
    phi[np.isnan(phi)] = 0.
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    
    # Convert to tensor
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    dist_ca = torch.from_numpy(dist_ca).to(dtype=torch.float32, device=device)
    omega = torch.from_numpy(omega).to(dtype=torch.float32, device=device)
    theta = torch.from_numpy(theta).to(dtype=torch.float32, device=device)
    dihedral = torch.from_numpy(dihedral).to(dtype=torch.float32, device=device)
    phi = torch.from_numpy(phi).to(dtype=torch.float32, device=device)
    mask_angle = torch.from_numpy(mask_angle).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    #     mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, chain_M, residue_idx, chain_encoding_all


#residue_idx = residue idx with jumps across chains
#chain_M = 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
#mask_self = for interface loss calculation - 0.0 for self interaction, 1.0 for other
#chain_encoding_all = integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
#S = sequence AAs integers

def make_block_diagonal(matrices, L_max):
    block_diagonal = np.zeros((L_max, L_max))
    # Track the starting indices for each matrix in the block diagonal
    start_index = 0

    # Concatenate the matrices in a block diagonal manner
    for matrix in matrices:
        matrix_size = matrix.shape[0]
        block_diagonal[start_index:start_index+matrix_size, start_index:start_index+matrix_size] = matrix
        start_index += matrix_size
    return block_diagonal