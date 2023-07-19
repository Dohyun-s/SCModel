import numpy as np
import random
import torch
from minsu_mpnn.data import Alphabet
from minsu_mpnn.custom_types import Batch

alphabet = Alphabet.from_architecture('ESM-1b')
truncation_seq_length = None

def tokenize_batch_sequences(batch_sequences, alphabet, truncation_seq_length):
    batch_size = len(batch_sequences)
    seq_str_list = [x for x in batch_sequences]
    seq_encoded_list = [alphabet.encode(seq_str) for seq_str in seq_str_list]
    if truncation_seq_length:
        seq_encoded_list = [seq_str[:truncation_seq_length] for seq_str in seq_encoded_list]

    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    tokens = torch.empty(
            (
                batch_size,
                max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
            ),
            dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)

    for i, seq_encoded in enumerate(seq_encoded_list):
        if alphabet.prepend_bos:
            tokens[i,0] = alphabet.cls_idx

        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        tokens[
            i,
            int(alphabet.prepend_bos):len(seq_encoded)+int(alphabet.prepend_bos),
        ] = seq

        if alphabet.append_eos:
            tokens[i, len(seq_encoded)+int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens

def collator(items):
    NUM_ATOM_TYPES = 4
    XYZ_DIM = 3
    items = list(filter(lambda x: x is not None, items))
    if len(items) == 0:
        return None  

    # sequence_hashes = [item['seq_hash'] for item in items]
    batch_size = len(items)
    sequence_lengths = np.array([len(item['seq']) for item in items])
    max_sequence_length = np.max(sequence_lengths)

    X = np.zeros([batch_size, max_sequence_length, NUM_ATOM_TYPES, XYZ_DIM], dtype=np.float32)

    residue_idx = -100 * np.ones([batch_size, max_sequence_length], dtype=np.int32)
    chain_M = np.zeros([batch_size, max_sequence_length], dtype=np.int32)
    mask_self = np.zeros([batch_size, max_sequence_length, max_sequence_length], dtype=np.int32)

    chain_encoding_all = np.zeros([batch_size, max_sequence_length], dtype=np.int32)

    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet

    batch_sequences = [item['seq'] for item in items]
    tokenized_batch_sequences = tokenize_batch_sequences(batch_sequences, alphabet, truncation_seq_length)

    for i, item in enumerate(items):
        masked_chains = item['masked_list']
        visible_chains = item['visible_list']
        all_chains = masked_chains + visible_chains

        visible_dict = {}
        masked_dict = {}

        for j, chain_letter in enumerate(all_chains):
            chain_seq = item[f'seq_chain_{chain_letter}']
            if chain_letter in visible_chains:
                visible_dict[chain_letter] = chain_seq
            elif chain_letter in masked_chains:
                masked_dict[chain_letter] = chain_seq

        for masked_letter, masked_seq in masked_dict.items():
            for visible_letter, visible_seq in visible_dict.items():
                if masked_seq == visible_seq:
                    if visible_letter not in masked_chains:
                        masked_chains.append(visible_letter)
                    if visible_letter in visible_chains:
                        visible_chains.remove(visible_letter)

        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)
        num_chains = item['num_of_chains']
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []

        c = 1
        chain_start_index = 0
        chain_end_index = 0

        for step, chain_letter in enumerate(all_chains):
            chain_seq = item[f'seq_chain_{chain_letter}']
            chain_length = len(chain_seq)
            chain_coords = item[f'coords_chain_{chain_letter}']
            if chain_letter in visible_chains:
                chain_mask = np.zeros(chain_length)
            if chain_letter in masked_chains:
                chain_mask = np.ones(chain_length)

            x_chain = np.stack(
                [
                    chain_coords[c] for c in 
                    [
                        f'N_chain_{chain_letter}', 
                        f'CA_chain_{chain_letter}', 
                        f'C_chain_{chain_letter}', 
                        f'O_chain_{chain_letter}',
                    ]
                ],
                axis=1,
            )
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(
                c*np.ones(np.array(chain_mask).shape[0])
            )
            chain_end_index += chain_length
            mask_self[i, chain_start_index:chain_end_index, chain_start_index:chain_end_index] = \
                    np.zeros([chain_length, chain_length])
            residue_idx[i, chain_start_index:chain_end_index] = 100*(c-1) + np.arange(chain_start_index, chain_end_index)
            chain_start_index += chain_length
            c += 1

        x = np.concatenate(x_chain_list, 0)
        all_sequence = "".join(chain_seq_list)
        mask = np.concatenate(chain_mask_list, 0)
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        all_sequence_length = len(all_sequence)
        x_pad = np.pad(x, [[0,max_sequence_length-all_sequence_length], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        mask_pad = np.pad(mask, [[0,max_sequence_length-all_sequence_length]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,max_sequence_length-all_sequence_length]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

    is_nan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[is_nan] = 0

    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)
    return Batch(X, tokenized_batch_sequences, mask, sequence_lengths, chain_M, residue_idx, mask_self, chain_encoding_all)
