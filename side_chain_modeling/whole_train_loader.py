import sys
sys.path.append("..")
from ProteinMPNN.training.utils import build_training_clusters, StructureDataset, StructureLoader, \
                        PDB_dataset, loader_pdb, worker_init_fn
from ProteinMPNN.training.model_utils import get_std_opt, get_scheduler

import torch
import numpy as np
import logging
import numpy as np
import time
import torch
import scipy
import scipy.spatial
import torch.nn as nn
import torch.nn.functional as F
import wandb
import gc
import queue
import time
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)

from util import get_coords6d, _dihedrals, _normalize, get_pdbs, generate_Cbeta,\
    get_dihedrals, get_angles, featurize
from util_module import XYZConverter
from model import ProteinMPNN
from loss import torsionAngleLoss

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
train_set = PDB_dataset(list(train.keys()), loader_pdb, train, PARAMS)
train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, PARAMS)
valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)


num_examples_per_epoch=5000

hidden_dim=128
max_protein_length = 5000
batch_size = 5000


from concurrent.futures import ProcessPoolExecutor    

with ProcessPoolExecutor(max_workers=12) as executor:
    q = queue.Queue(maxsize=3)
    p = queue.Queue(maxsize=3)

    for i in range(3):
        q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
        p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))
    pdb_dict_train = q.get().result()
    print('fist')
    pdb_dict_valid = p.get().result()
    print('second')
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length)
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)
    print('third')
    loader_train = StructureLoader(dataset_train, batch_size=batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)
    print('fourth')

import pickle
with open('loader_train2.pkl', 'wb') as file:
    pickle.dump(loader_train, file)
with open('loader_valid2.pkl', 'wb') as file:
    pickle.dump(loader_valid, file)
