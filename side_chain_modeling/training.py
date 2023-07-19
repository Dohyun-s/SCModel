import sys
sys.path.append("..")
from ProteinMPNN.training.utils import build_training_clusters, StructureDataset, StructureLoader, \
                        PDB_dataset, loader_pdb, worker_init_fn
import torch
import numpy as np
import logging
import numpy as np
import time
import torch
import scipy
import scipy.spatial

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)

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

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres),999.9, dtype=np.float32)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres), dtype=np.float32)
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres), dtype=np.float32)
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres), dtype=np.float32)
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d

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
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = torch.Tensor(t['xyz'][res,])[0,] #[L, 14, 3]
                            try:
                                dist6d, omega6d, theta6d, phi6d = get_coords6d(all_atoms, 20.0)
                            except Exception:
                                continue
                            dbins = np.linspace(2.5, 20.0, 36)
                            d6d = np.digitize(dist6d, dbins)
                            coords_dict_chain['dist'+letter]=dbins
                            coords_dict_chain['omega'+letter]=omega6d
                            coords_dict_chain['theta'+letter]=dist6d
                            coords_dict_chain['phi'+letter]=phi6d
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

LOCAL_PATH = "/home/minsu/CLIPP/training_data/pdb_2021aug02_sample"
DATA_PATH = "/public_data/ml/RF2_train/PDB-2021AUG02"
MY_LOCAL = "/home/dohyun/project/"
PARAMS = {
    "LIST"    : f"{MY_LOCAL}/train_s",
    "VAL"     : f"{DATA_PATH}/PDB_val",
    "TEST"    : f"{LOCAL_PATH}/test_clusters.txt",
    "STRUCT_CLUST" : f"{LOCAL_PATH}/seq_hash_to_clust_hash.yaml",  
    "DIR"     : f"{DATA_PATH}/torch",
    "DATCUT"  : "2030-Jan-01",
    "RESCUT"  : 3.5,
    "HOMO"    : 0.70, # min sequence identity for homologous chains
    "CHAIN_ONLY": False,
    "HARD"    : f"{LOCAL_PATH}/hard_negative.yaml",
}
logging.info("LOADING DATA")
LOAD_PARAM = {'batch_size': 1,
              'shuffle': True,
              'pin_memory':False,
              'num_workers': 4}
train, valid, test = build_training_clusters(PARAMS, False)
train_set = PDB_dataset(list(train.keys()), loader_pdb, train, PARAMS)
train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, PARAMS)
valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
# train_assemblies = PDBAssemblies(training_clusters.train, training_clusters.chainid_to_hash,
#                                  PARAMS)
# valid_assemblies = PDBAssemblies(training_clusters.valid, training_clusters.chainid_to_hash,
#                                  PARAMS)

# toks_per_batch=5000
# logging.info(len(train_assemblies))
# train_loader = torch.utils.data.DataLoader(
#         train_assemblies,
#         worker_init_fn=np.random.seed(),
#         collate_fn=collator,
#         batch_sampler=train_assemblies.get_batch_indices(
#             toks_per_batch=toks_per_batch, 
#             extra_toks_per_seq=2),
#         num_workers=16,
# )
# valid_loader = torch.utils.data.DataLoader(
#         valid_assemblies,
#         worker_init_fn=np.random.seed(),
#         collate_fn=collator,
#         batch_sampler=valid_assemblies.get_batch_indices(toks_per_batch=toks_per_batch,
#                                                          extra_toks_per_seq=2),
#         num_workers=16,
# )

from ProteinMPNN.training.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
import queue
import time
num_examples_per_epoch=50

hidden_dim=128
num_encoder_layers=3
num_neighbors=48
dropout=0.1
backbone_noise=0.2
max_protein_length = 10000
batch_size = 10000
reload_data_every_n_epochs = 2
mixed_precision = True
epoch = 0
gradient_norm = 1.0
scaler = torch.cuda.amp.GradScaler()
logfile = 'log.txt'
model = ProteinMPNN(node_features=hidden_dim, 
                        edge_features=hidden_dim, 
                        hidden_dim=hidden_dim, 
                        num_encoder_layers=num_encoder_layers, 
                        num_decoder_layers=num_encoder_layers, 
                        k_neighbors=num_neighbors, 
                        dropout=dropout, 
                        augment_eps=backbone_noise)
model.to(device)
total_step = 0
optimizer = get_std_opt(model.parameters(), hidden_dim, total_step)

from concurrent.futures import ProcessPoolExecutor    

with ProcessPoolExecutor(max_workers=12) as executor:
    q = queue.Queue(maxsize=3)
    p = queue.Queue(maxsize=3)
    get_pdbs(train_loader, 1, max_protein_length, num_examples_per_epoch)

    for i in range(3):
        q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
        p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))
    pdb_dict_train = q.get().result()
    pdb_dict_valid = p.get().result()
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)

    loader_train = StructureLoader(dataset_train, batch_size=batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)
    import sys
    sys.exit(1)
#     reload_c = 0 
    for e in range(10):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.
#         if e % reload_data_every_n_epochs == 0:
#             if reload_c != 0:
#                 pdb_dict_train = q.get().result()
#                 dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length)
#                 loader_train = StructureLoader(dataset_train, batch_size=batch_size)
#                 pdb_dict_valid = p.get().result()
#                 dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)
#                 loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)
#                 q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
#                 p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))
#             reload_c += 1
        for _, batch in enumerate(loader_train):
            start_batch = time.time()
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M

            if mixed_precision:
                with torch.cuda.amp.autocast():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                scaler.scale(loss_av_smoothed).backward()

                if gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                loss_av_smoothed.backward()

                if gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                optimizer.step()

            loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            total_step += 1

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            for _, batch in enumerate(loader_valid):
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                mask_for_loss = mask*chain_M
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_accuracy = validation_acc / validation_weights
        validation_perplexity = np.exp(validation_loss)

        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
        validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
