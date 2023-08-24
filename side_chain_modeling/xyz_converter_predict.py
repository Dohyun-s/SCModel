import torch
import numpy as np
from util_module import XYZConverter
import sys
sys.path.append("..")
from ProteinMPNN.training.utils import build_training_clusters, StructureDataset, StructureLoader, \
                        PDB_dataset, loader_pdb, worker_init_fn
from ProteinMPNN.training.model_utils import get_std_opt, get_scheduler

import logging

# # full sc atom representation (Nx14)
# aa2long=[
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
#     (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
#     (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
#     (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
#     (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
#     (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
#     (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
#     (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
#     (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
#     (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
#     (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
#     (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
#     (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
# ]

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # mask
]



def write_pdb(seq, atoms, Bfacts=None, prefix=None, chainlen=None):
    L = len(seq)
    if chainlen is None:
        chainlen = L

    filename = "output/%s.pdb"%prefix
    ctr = 1
    with open(filename, 'wt') as f:
        if Bfacts == None:
            Bfacts = np.zeros(L)
        else:
            Bfacts = torch.clamp( Bfacts, 0, 1)

        chains="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz01234567"
        for i,s in enumerate(seq):
            i_seq = (i%chainlen)+1
            chn = chains[i//chainlen]
            if (len(atoms.shape)==2):
                if (not torch.any(torch.isnan(atoms[i]))):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr%100000, " CA ", num2aa[s], 
                            chn, i_seq, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    ctr += 1

            elif atoms.shape[1]==3:
                for j,atm_j in enumerate((" N  "," CA "," C  ")):
                    if (not torch.any(torch.isnan(atoms[i,j]))):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr%100000, atm_j, num2aa[s], 
                                chn, i_seq, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1
            else:
                atms = aa2long[s]
                for j,atm_j in enumerate(atms):
                    if (atm_j is not None):
                        if (not torch.any(torch.isnan(atoms[i,j]))):
                            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr%100000, atm_j, num2aa[s], 
                                chn, i_seq, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                            ctr += 1


use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)

from util import get_pdbs, featurize

LOCAL_PATH = "/home/minsu/CLIPP/training_data/pdb_2021aug02_sample"
DATA_PATH = "/public_data/ml/RF2_train/PDB-2021AUG02"
MY_LOCAL = "/home/dohyun/project/"
PARAMS = {
    "LIST"    : f"{LOCAL_PATH}/list.csv",
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
test_set = PDB_dataset(list(test.keys()), loader_pdb, test, PARAMS)
test_loader = torch.utils.data.DataLoader(test_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

max_protein_length = 10000
batch_size = 10000

num_examples_per_epoch=1
hidden_dim=128
num_encoder_layers=3
num_neighbors=32
dropout=0.1
backbone_noise=0.2
max_protein_length = 10000
batch_size = 10000
reload_data_every_n_epochs = 2
mixed_precision = True
epoch = 0
gradient_norm = 1.0
scaler = torch.cuda.amp.GradScaler()
from model import ProteinMPNN
model = ProteinMPNN(node_features=hidden_dim, 
                        edge_features=hidden_dim, 
                        hidden_dim=hidden_dim, 
                        num_encoder_layers=num_encoder_layers, 
                        num_decoder_layers=num_encoder_layers, 
                        k_neighbors=num_neighbors, 
                        dropout=dropout, 
                        augment_eps=backbone_noise)
checkpoint = torch.load("model_default_loss.pt", map_location=torch.device('cpu'))

# Extract the model from the dictionary (assuming the key is 'model')
model.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the desired device
device = torch.device('cpu')  # Replace 'cpu' with 'cuda' if you have a GPU
model.to(device)


pdb_dict_test = get_pdbs(test_loader, 1, max_protein_length, num_examples_per_epoch)

dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=max_protein_length) 

loader_test = StructureLoader(dataset_test, batch_size=batch_size)
for test_batch in loader_test:

    dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, chain_M, residue_idx,\
                                chain_encoding_all = featurize(test_batch, device)
    with torch.cuda.amp.autocast():
        result = model(dist_ca, omega, theta, phi, dihedral, mask_angle, mask, \
                        S, chain_M, residue_idx, chain_encoding_all)
    break

#     predict(model, result)

n_recycles = 3
from util_module import XYZConverter

alphabet = list('ARNDCQEGHILKMFPSTWYVX-')
char_to_num = {char: num for num, char in enumerate(alphabet)}
seq = [char_to_num[char] for char in test_batch[0]['seq_chain_A']]
seq = torch.unsqueeze(torch.Tensor(seq), 0).long()

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]

xyz_prev = torch.unsqueeze(test_batch[0]['coords_chain_A']['xyz_coords'], 0)
write_pdb(seq[0], xyz_prev[0], prefix=test_batch[0]['name'])
with torch.cuda.amp.autocast(True):
    # _, best_xyz = XYZConverter().compute_all_atom(seq, xyz_prev, result[:,:len(test_batch[0]['seq_chain_A'])], use_H=False)
    from util_module import ComputeAllAtomCoords
    _, best_xyz = ComputeAllAtomCoords().forward(seq, xyz_prev, result[:,:len(test_batch[0]['seq_chain_A'])], use_H=False)
best_xyz = best_xyz.float().cpu()
import pdb
pdb.set_trace()
write_pdb(seq[0], best_xyz[0], prefix=test_batch[0]['name']+'_predict')

# import os
# import glob

# pattern = '/output/*.pdb'

# files_to_delete = glob.glob(pattern)
# for file_path in files_to_delete:
#     try:
#         os.remove(file_path)
#         print(f"Deleted: {file_path}")
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")