import torch

class Batch:
    def __init__(self, X, tokenized_batch_sequences, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all):
        self.X = X
        self.tokenized_batch_sequences = tokenized_batch_sequences
        self.mask = mask
        self.lengths = lengths
        self.chain_M = chain_M
        self.residue_idx = residue_idx
        self.mask_self = mask_self
        self.chain_encoding_all = chain_encoding_all

    def to(self, device):
        self.X = self.X.to(dtype=torch.float32, device=device)
        self.tokenized_batch_sequences = self.tokenized_batch_sequences.to(dtype=torch.long, device=device)
        self.mask = self.mask.to(dtype=torch.float32, device=device)
        self.chain_M = self.chain_M.to(dtype=torch.float32, device=device)
        self.residue_idx = self.residue_idx.to(dtype=torch.long, device=device)
        self.mask_self = self.mask_self.to(dtype=torch.float32, device=device)
        self.chain_encoding_all = self.chain_encoding_all.to(dtype=torch.long, device=device)

    def __len__(self):
        return self.X.shape[0] # X.shape = (batch_size, max_sequence_length, NUM_ATOM_TYPES, XYZ_DIM)


class TrainingClusters:
    def __init__(
        self, 
        train, 
        valid, 
        test, 
        chainid_to_hash
    ):
        self.train = train  
        self.valid = valid
        self.test = test
        self.chainid_to_hash = chainid_to_hash
