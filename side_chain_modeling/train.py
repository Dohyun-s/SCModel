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
wandb.init(
    # set the wandb project where this run will be logged
    project="side_chain_modeling",
    # track hyperparameters and run metadata
    config={
    "epochs": 200,
    }, resume="allow", allow_val_change=True
)

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


num_examples_per_epoch=50

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
model = ProteinMPNN(node_features=hidden_dim, 
                        edge_features=hidden_dim, 
                        hidden_dim=hidden_dim, 
                        num_encoder_layers=num_encoder_layers, 
                        num_decoder_layers=num_encoder_layers, 
                        k_neighbors=num_neighbors, 
                        dropout=dropout, 
                        augment_eps=backbone_noise)
model.to(device)
wandb.watch(model)

total_step = 0
optimizer = get_std_opt(model.parameters(), hidden_dim, total_step)
num_warmup_steps = 4000
scheduler = get_scheduler(optimizer.optimizer, warmup_steps=num_warmup_steps)

from concurrent.futures import ProcessPoolExecutor    

with ProcessPoolExecutor(max_workers=12) as executor:
    q = queue.Queue(maxsize=3)
    p = queue.Queue(maxsize=3)

    for i in range(3):
        q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
        p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))
    pdb_dict_train = q.get().result()
    pdb_dict_valid = p.get().result()
    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length) 
    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)

    loader_train = StructureLoader(dataset_train, batch_size=batch_size)
    loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)

    wandb.config.train_dataset_length = len(loader_train)
    wandb.config.val_dataset_length = len(loader_valid)
    print("\ttrain data len:", wandb.config.train_dataset_length)
    print("\tval data len:", wandb.config.val_dataset_length)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 20

    for e in range(250):
        t0 = time.time()
        e = epoch + e
        train_avg_loss = 0.0
        model.train()
        start_batch = time.time()
        for _, batch in enumerate(loader_train):
            dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, chain_M, residue_idx,\
                                chain_encoding_all = featurize(batch, device)
            optimizer.zero_grad()
            alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
            tors = []
            for s in range(len(batch)):
                all_chains = batch[s]['visible_list']+batch[s]['masked_list']
                coord = torch.cat([batch[s][f'coords_chain_{letter}']['xyz_coords'] for letter in all_chains])
                all_sequence = batch[s]['seq']
                indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
                seq_aa = indices
                true_tors, true_tors_alt, tors_mask, tors_planar = XYZConverter().get_torsions(
                            torch.unsqueeze(coord,0), 
                            torch.unsqueeze(torch.from_numpy(seq_aa),0).to(dtype=torch.long)
                )
                tors.append([true_tors.to(device), true_tors_alt.to(device), tors_mask.to(device), tors_planar.to(device)])
            
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    result = model(dist_ca, omega, theta, phi, dihedral, mask_angle, mask, \
                                    S, chain_M, residue_idx, chain_encoding_all)
                    l_tors_sum = 0
                    for s in range(len(batch)):
                        nres = len(batch[s]['seq'])
                        true_tors, true_tors_alt, tors_mask, tors_planar = tors[s]
                        l_tors = torsionAngleLoss(result[s][:nres].unsqueeze(0), true_tors, true_tors_alt, \
                                                    tors_mask, tors_planar, eps = 1e-10)
                        l_tors_sum += l_tors

                scaler.scale(l_tors_sum).backward()

                if gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_avg_loss += l_tors.detach()
            else:
                result = model(dist_ca, omega, theta, phi, dihedral, mask_angle, mask, \
                                S, chain_M, residue_idx, chain_encoding_all)
                l_tors_sum = 0
                for s in range(len(batch)):
                    nres = len(batch[s]['seq'])
                    true_tors, true_tors_alt, tors_mask, tors_planar = tors[s]
                    l_tors = torsionAngleLoss(result[s][:nres].unsqueeze(0), true_tors, true_tors_alt, \
                                                tors_mask, tors_planar, eps = 1e-10)
                    l_tors_sum += l_tors
                l_tors_sum.backward()

                if gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
                optimizer.step()
                scheduler.step()
                train_avg_loss += l_tors_sum.detach()
            
        elapsed_featurize = time.time() - start_batch
        
        train_avg_loss = train_avg_loss / len(loader_train)
        print ("Train epoch{}, time {:.2f}, loss {} ".format(e, elapsed_featurize, train_avg_loss.item()))

        model.eval()
        val_avg_loss = 0.0
        with torch.no_grad():
            for _, batch in enumerate(loader_valid):
                dist_ca, omega, theta, phi, dihedral, mask_angle, mask, S, chain_M, residue_idx,\
                                chain_encoding_all = featurize(batch, device)
                result = model(dist_ca, omega, theta, phi, dihedral, mask_angle, mask, \
                                    S, chain_M, residue_idx, chain_encoding_all)
                tors = []
                for s in range(len(batch)):
                    all_chains = batch[s]['visible_list']+batch[s]['masked_list']
                    coord = torch.cat([batch[s][f'coords_chain_{letter}']['xyz_coords'] for letter in all_chains])
                    all_sequence = batch[s]['seq']
                    indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
                    seq_aa = indices
                    true_tors, true_tors_alt, tors_mask, tors_planar = XYZConverter().get_torsions(
                                torch.unsqueeze(coord,0), 
                                torch.unsqueeze(torch.from_numpy(seq_aa),0).to(dtype=torch.long)
                    )
                    tors.append([true_tors.to(device), true_tors_alt.to(device), tors_mask.to(device), tors_planar.to(device)])
                
                l_tors_sum = 0
                for s in range(len(batch)):
                    nres = len(batch[s]['seq'])
                    true_tors, true_tors_alt, tors_mask, tors_planar = tors[s]
                    l_tors = torsionAngleLoss(result[s][:nres].unsqueeze(0), true_tors, true_tors_alt, \
                                                tors_mask, tors_planar, eps = 1e-10)
                    l_tors_sum += l_tors
                val_avg_loss += l_tors_sum.detach()
        val_avg_loss = val_avg_loss / len(loader_valid)
        print ("valid epoch {}, loss {} ".format(e, val_avg_loss.item()))
        
        wandb.log({"step": e,
                   "train_loss": train_avg_loss,
                   "val_loss": val_avg_loss,
                   }
                  )
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_without_improvement = 0
            torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    'train_loss': train_avg_loss,
                    'val_loss': val_avg_loss,
                    }, 'model_default_loss.pt')
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {e + 1}!")
            break
        gc.collect()