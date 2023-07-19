import itertools
import os
from typing import Sequence, Tuple, List, Union, Dict
from minsu_mpnn.constants import proteinseq_toks
from minsu_mpnn.custom_types import TrainingClusters

import torch
from dateutil import parser
import re
import random
import csv
import numpy as np
import time
import yaml

RawMSA = Sequence[Tuple[str, str]]
random.seed(42)

class PDBAssemblies(torch.utils.data.Dataset):
    """
    Dataset class for PDB Assemblies
    Loads PDB metadata, load corresponding chains and generates an assembly.

    Parameters
    ----------
    data_split_dict: Dict{Sequence Cluster ID: [[pdbid_chainid1, sequence_hash1], [pdbid_chainid2, sequence_hash2], ...]}
        corresponding to the data split (train, valid, test)
    PARAMS : Dict
        paths to list, val, dict files, data directory, data/res cut.

    Methods
    -------
    _get_assembly(item:[chain_id, sequence_hash], PARAMS)
        Given a single chain, load its corresponding PDB metadata and generate an assembly 
        chain_id = pdb_id + chain_letter
        returns a dictionary containing {seq, xyz, idx, label} of assembly

    _build_assembly(assembly)
        Given an assembly, build the assembly by applying the transformation matrices

    _parse_assembly(assembly)
        Given an assembly, parse the assembly into a sequence and a set of coordinates

    """
    def __init__(self, data_split_dict, chainid_to_hash, PARAMS):
        self.sequence_cluster_ids = list(data_split_dict.keys())
        self.data_split_dict = data_split_dict
        self.chainid_to_hash = chainid_to_hash

        self.PARAMS = PARAMS

        init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        extra_alphabet = [str(item) for item in list(np.arange(300))]

        self.chain_alphabet = init_alphabet + extra_alphabet

    def __len__(self):
        return len(self.sequence_cluster_ids)

    def __getitem__(self, index):
        # index[0] == sequence_cluster
        # index[1] == chain_id_idx
        sequence_cluster = self.data_split_dict[index[0]]
        chain_id = sequence_cluster[index[1]][0]

        if not self.PARAMS["CHAIN_ONLY"]:
            return self._get_assembly(chain_id)
        return self._get_chains_separately(chain_id)

    def _parse_assembly(
        self, 
        assembly, 
        max_length=10000, 
    ):
        """
        Function for parsing given assemblies.

        Given an assembly loader, iterate through all assemblies and divide into:
        - seq_chain_{chain_letter}
        - coords_chain_{chain_letter}
        - name
        - chain_list
        - num_of_chains
        - seq (of the whole protein)

        Parameters
        ----------
        assembly : dict
            Dictionary containing the assembly data. Assembly is build by _build_assembly and processed in _get_assembly
        max_length : int, optional
            Maximum length allowed for the concatenated sequence, by default 10000.

        Returns
        -------
        dict or None
            Parsed assembly data as a dictionary if the sequence length is within the limit,
            otherwise None.

        """ 

        assembly = {k:v for k,v in assembly.items()}
        if 'label' in list(assembly):
            temp_dict = {}
            concat_seq = ''
            coords_dict = {}
            masked_list = []
            visible_list = []
            if len(list(np.unique(assembly['idx']))) < 352:
                # for every chain
                for idx in list(np.unique(assembly['idx'])):
                    letter = self.chain_alphabet[idx]
                    res = np.argwhere(assembly['idx']==idx)
                    initial_sequence = "".join(
                        list(np.array(list(assembly['seq']))[res][0,])
                    )
                    res = self._check_histidine_tag(initial_sequence, res)
                    if res.shape[1] < 4:
                        continue

                    temp_dict['seq_chain_'+letter]= "".join(
                        list(np.array(list(assembly['seq']))[res][0,])
                    )
                    concat_seq += temp_dict['seq_chain_'+letter]

                    if idx in assembly['masked']:
                        masked_list.append(letter)
                    else:
                        visible_list.append(letter)

                    coords_dict_chain = {}
                    all_atoms = np.array(assembly['xyz'][res,])[0,] #[L, 14, 3]
                    coords_dict_chain['N_chain_'+letter]=all_atoms[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_'+letter]=all_atoms[:, 1, :].tolist()
                    coords_dict_chain['C_chain_'+letter]=all_atoms[:, 2, :].tolist()
                    coords_dict_chain['O_chain_'+letter]=all_atoms[:, 3, :].tolist()
                    temp_dict['coords_chain_'+letter]=coords_dict_chain

                temp_dict['name']= assembly['label']
                temp_dict['masked_list'] = masked_list
                temp_dict['visible_list']= visible_list
                temp_dict['num_of_chains'] =  len(masked_list) + len(visible_list)
                temp_dict['seq'] = concat_seq
                # TODO This needs to be handled for complexes as well.
                # parse_assembly will not work for complexes, only for single chains
                temp_dict['seq_hash'] = assembly['seq_hash']

                # TODO: If we are going to train by chain, this is unnecessary.
                # Check if sequence is too long above and continue if it is too long
                if len(concat_seq) <= max_length:
                    return temp_dict
                else:
                    print('Sequence too long')
                    return None
        return None

    def _check_histidine_tag(self, initial_sequence, res):
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
        return res

    def _build_assembly(self, idx, meta, asmb_chains, chains):
        asmb = {}
        for k in idx:
            # pick kth xform
            xform = meta[f'asmb_xform{k}']
            u = xform[:, :3, :3]
            r = xform[:, :3, 3]

            # select chains which k-th xform should be applied to
            s1 = set(meta['chains'])
            s2 = set(asmb_chains[k].split(','))
            chains_k = s1 & s2

            # transform selected chains
            for c in chains_k:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i, xyz_i in enumerate(xyz_ru)})
        return asmb

    def _get_chains_separately(self, chain_id):
        sequence_hash = self.chainid_to_hash[chain_id]

        pdb_id, chain_letter = chain_id.split('_')
        PREFIX = f"{self.PARAMS['DIR']}/pdb/{pdb_id[1:3]}/{pdb_id}"
        
        # load metadata
        if not os.path.isfile(PREFIX+".pt"):
            assembly = {
                'seq': np.zeros(5)
            }
            return self._parse_assembly(assembly)

        chain = torch.load(f"{PREFIX}_{chain_letter}.pt")
        sequence_length = len(chain['seq'])

        assembly = {
                'seq'     : chain['seq'],
                'seq_hash': sequence_hash,
                'xyz'     : chain['xyz'],
                'idx'     : torch.zeros(sequence_length).int(),
                'masked'  : torch.Tensor(0).int(),
                'label'   : chain_id,
        }
        return self._parse_assembly(assembly)

    def _get_assembly(self, chain_id):
        #TODO If we are going to use biological assembly structures for training,
        # we need to add sequence_hashes to the returning assembly to get tm scores.
        pdb_id, chain_letter = chain_id.split('_')
        PREFIX = f"{self.PARAMS['DIR']}/pdb/{pdb_id[1:3]}/{pdb_id}"
        
        # load metadata
        if not os.path.isfile(PREFIX+".pt"):
            assembly = {
                'seq': np.zeros(5)
            }
            return self._parse_assembly(assembly)

        meta = torch.load(PREFIX+".pt") 
        asmb_ids = meta['asmb_ids']
        asmb_chains = meta['asmb_chains']
        chain_letters = np.array(meta['chains'])

        # find candidate assemblies which contain chain_letter chain
        asmb_candidates = set(
            [asmb_id for asmb_id, asmb_chain in zip(asmb_ids,asmb_chains)
            if chain_letter in asmb_chain.split(',')]
        )

        # if the chains is missing from all the assemblies
        # then return this chain alone
        if len(asmb_candidates) < 1:
            chain = torch.load(f"{PREFIX}_{chain_letter}.pt")
            sequence_length = len(chain['seq'])

            assembly = {
                    'seq'    : chain['seq'],
                    'xyz'    : chain['xyz'],
                    'idx'    : torch.zeros(sequence_length).int(),
                    'masked' : torch.Tensor(0).int(),
                    'label'  : chain_id,
            }
            return self._parse_assembly(assembly)

        # randomly pick one assembly from candidates
        asmb_i = random.sample(list(asmb_candidates), 1)
        # indices of selected transforms
        idx = np.where(np.array(asmb_ids)==asmb_i)[0]
        # load relevant chains
        assembly_chains = {
            chain:torch.load(f"{PREFIX}_{chain}.pt")
            for i in idx for chain in asmb_chains[i]
            if chain in meta['chains']
        }

        # build assembly
        try: 
            asmb = self._build_assembly(idx, meta, asmb_chains, assembly_chains)
        except KeyError:
            assembly = {
                'seq': np.zeros(5)
            }   
            return self._parse_assembly(assembly)

        # select chains which share considerable similarity to chain_letter
        sequence_identity = meta['tm'][chain_letters==chain_letter][0,:,1]
        homo = set(
                    [
                        chain_j for sequence_identity_j, chain_j in zip(sequence_identity, chain_letters)
                        if sequence_identity_j > self.PARAMS['HOMO']
                    ]
                )
        # stack all chains in the assembly together
        # asmb idx are the indices of each chain
        asmb_seq, asmb_xyz, asmb_idx, masked = "", [], [], []
        for counter, (asmb_key, asmb_val) in enumerate(asmb.items()):
            asmb_seq += assembly_chains[asmb_key[0]]['seq']
            asmb_xyz.append(asmb_val)
            asmb_idx.append(torch.full((asmb_val.shape[0],),counter))
            if asmb_key[0] in homo:
                masked.append(counter)

        assembly = {
                'seq'    : asmb_seq,
                'xyz'    : torch.cat(asmb_xyz,dim=0),
                'idx'    : torch.cat(asmb_idx,dim=0),
                'masked' : torch.Tensor(masked).int(),
                'label'  : chain_id,
        }
        parsed = self._parse_assembly(assembly)
        return parsed

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq):
        batches = []
        buf = []
        max_len = 0
        overlapping_sequence_hashes = set()

        all_chains = []
        for sequence_cluster, chains in self.data_split_dict.items():
            # number_of_chains_per_seq_cluster = min(len(chains), 10)
            number_of_chains_per_seq_cluster = min(len(chains), 1)

            sampled_chain_indices = random.sample(range(len(chains)), number_of_chains_per_seq_cluster)
            sampled_chains = [chains[sampled_chain_idx] for sampled_chain_idx in sampled_chain_indices]

            for i, sampled_chain in enumerate(sampled_chains):
                # sampled_chain[0] = chain ID
                # sampled_chain[1] = sequence hash
                # sampled_chain[2] = sequence length
                sampled_chain_seq_hash = sampled_chain[1]
                if sampled_chain_seq_hash in overlapping_sequence_hashes:
                    continue
                else:
                    overlapping_sequence_hashes.add(sampled_chain_seq_hash)
                    # all chains -> (sequence_cluster, index_of_sampled_chain, chain_id, sequence_hash, sequence_length)
                    all_chains.append((sequence_cluster, sampled_chain_indices[i], sampled_chain[0], sampled_chain[1], sampled_chain[2]))

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                 return
            batches.append(buf)
            buf = []
            max_len = 0

        # argsort all sampled chains by chain length
        chain_sizes = np.array([chain[4] for chain in all_chains])
        argsort = np.argsort(chain_sizes)

        for sort_idx in argsort:
            chain = all_chains[sort_idx]
            sequence_cluster = chain[0]
            size = chain[4] + extra_toks_per_seq

            if max(size, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, size)
            # we index the dataloader by (sequence_cluster, index of sampled chain)
            buf.append((sequence_cluster, chain[1]))

        _flush_current_buf()

        # for feature caching, it might be beneifical to randomly shuffle the batch orders
        # to introduce diverse protein sizes into the "contrastive matrix".
        # we shuffle the batches list to achieve this. 
        random.shuffle(batches)
        return batches

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
            return BatchConverter(self, truncation_seq_length)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


def build_training_clusters(PARAMS):
    val_ids = set([int(l) for l in open(PARAMS['VAL']).readlines()])
    test_ids = set([int(l) for l in open(PARAMS['TEST']).readlines()])
 
    # read & clean list.csv
    with open(PARAMS['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # read[0] = chain ID
        # read[1] = deposition date
        # read[2] = resolution
        # read[3] = sequence hash
        # read[4] = sequence cluster
        # read[5] = sequence
        rows = [[read[0], read[3], int(read[4]), len(read[5])] for read in reader
                if float(read[2])<=PARAMS['RESCUT'] and
                parser.parse(read[1])<=parser.parse(PARAMS['DATCUT'])]
    
    # compile training and validation sets
    # Dict{Sequence Cluster : [(chain_id_1, sequence_hash_1, sequence_length_1), (chain_id_2, sequence_hash_2, sequence_length_2) ...]}
    train = {}
    valid = {}
    test = {}

    chainid_to_hash = {}

    # rows = [row for row in rows if row[3] <= 500]
     
    for r in rows:
        # r[0] = chain ID
        # r[1] = sequence hash
        # r[2] = sequence cluster
        # r[3] = sequence length

        # chainid_to_hash is needed, as when we build
        # a biological assembly, we load other chains
        # in the same biological assembly.
        chainid_to_hash[r[0]] = r[1]

        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append((r[0], r[1], r[3]))
            else:
                valid[r[2]] = [(r[0],r[1],r[3])]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append((r[0], r[1], r[3]))
            else:
                test[r[2]] = [(r[0],r[1],r[3])]
        else:
            if r[2] in train.keys():
                train[r[2]].append((r[0], r[1], r[3]))
            else:
                train[r[2]] = [(r[0],r[1],r[3])]     

    return TrainingClusters(train, valid, test, chainid_to_hash)
