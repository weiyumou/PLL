import random

import torch
import tqdm
from torch.utils.data import Dataset


def calc_dn(n):
    """
    Calculate the number of derangement D_n
    :param n: The length of the sequence
    :return: A list of number for all 0 <= i <= n
    """
    ds = [1, 0]
    for i in range(2, n + 1):
        ds.append((i - 1) * (ds[i - 1] + ds[i - 2]))
    return ds


def random_derangement(n, ds):
    """
    Implementation of the algorithm for generating random derangement of length n,
    as described in the paper "Generating Random Derangements"
    retrieved at https://epubs.siam.org/doi/pdf/10.1137/1.9781611972986.7
    :param n: The length of the derangement
    :param ds: A list of lengths of derangement for all 0 <= i <= n
    :return: A random derangement
    """
    perm = list(range(n))
    mask = [False] * n

    i, u = n - 1, n - 1
    while u >= 1:
        if not mask[i]:
            j = random.randrange(i)
            while mask[j]:
                j = random.randrange(i)
            perm[i], perm[j] = perm[j], perm[i]
            p = random.random()
            if p < u * ds[u - 1] / ds[u + 1]:
                mask[j] = True
                u -= 1
            u -= 1
        i -= 1
    return perm


def k_permute(n, k, ds):
    """
    Produces a random permutation of n elements that contains a derangment of k elements
    :param n: Total number of elements
    :param k: The length of the derangement
    :param ds: A list of lengths of derangement for all 0 <= i <= n
    :return: A random permutation with a derangement of the desired length
    """
    k = min(k, n)
    indices = list(range(n))
    sel_indices = sorted(random.sample(indices, k))
    perm = random_derangement(k, ds)
    new_indices = indices.copy()
    for i, p in enumerate(perm):
        new_indices[sel_indices[i]] = indices[sel_indices[p]]
    return new_indices


def is_doc_start(line):
    return line.startswith("[")


class WikiReader:
    def __init__(self, data_file, lines_per_doc, num_lines=-1, train_perct=0.9):
        docs, seq = [], []
        curr_num_lines = 0
        with open(data_file, "r") as file_hdl:
            for line in tqdm.tqdm(file_hdl, desc="Reading File"):
                line = line.rstrip("\n")
                if not line:
                    continue
                if curr_num_lines == lines_per_doc:
                    if seq:
                        docs.append(seq)
                        seq = []
                        curr_num_lines = 0
                if is_doc_start(line):
                    seq = []
                    curr_num_lines = 0
                    continue
                seq.append(line)
                curr_num_lines += 1
                num_lines -= 1
                if num_lines == 0:
                    break

        train_idx = int(train_perct * len(docs))
        self.train_set = docs[:train_idx]
        self.val_set = docs[train_idx:]


class WikiDocDataset(Dataset):

    def __init__(self, docs, tokeniser, max_seq_len) -> None:
        super(WikiDocDataset, self).__init__()
        self.data = docs
        self.tokeniser = tokeniser
        self.max_seq_len = max_seq_len

    def __getitem__(self, index: int):
        token_ids, token_masks = [], []
        for sentence in self.data[index]:
            ids = torch.tensor(self.tokeniser.encode(sentence, max_length=self.max_seq_len), dtype=torch.long)
            ids = torch.nn.functional.pad(ids, [0, self.max_seq_len - ids.size(0)])
            mask = ids != 0
            token_ids.append(ids)
            token_masks.append(mask)

        token_ids = torch.stack(token_ids, dim=0)
        token_masks = torch.stack(token_masks, dim=0)
        return token_ids, token_masks

    def __len__(self) -> int:
        return len(self.data)


class WikiDataset(Dataset):
    def __init__(self, docs, tokeniser, max_seq_len, num_derangements):
        super(WikiDataset, self).__init__()
        self.data = WikiDocDataset(docs, tokeniser, max_seq_len)
        self.num_derangements = num_derangements
        self.ds = calc_dn(num_derangements)

    def _generate_derangement(self, n):
        sel_indices = list(sorted(random.sample(range(n), self.num_derangements)))
        perm = random_derangement(self.num_derangements, self.ds)
        return sel_indices, perm

    def __len__(self):
        return len(self.data)


class WikiTrainDataset(WikiDataset):
    def __getitem__(self, index: int):
        token_ids, token_masks = self.data[index]
        num_sents = token_ids.size(0)
        sel_indices, sent_perm = self._generate_derangement(num_sents)
        token_ids[sel_indices] = token_ids[sel_indices][sent_perm]
        token_masks[sel_indices] = token_masks[sel_indices][sent_perm]

        sent_perm = torch.tensor(sent_perm, dtype=torch.long)
        sel_indices = torch.tensor(sel_indices, dtype=torch.long)
        sent_position_id = torch.arange(num_sents).scatter_(0, sel_indices, value=num_sents)
        sent_type_id = torch.zeros(num_sents, dtype=torch.long).scatter_(0, sel_indices, value=1)
        return token_ids, token_masks, sent_position_id, sent_type_id, sent_perm


class WikiEvalDataset(WikiDataset):
    def __init__(self, eval_data, tokeniser, max_seq_len, num_derangements):
        super(WikiEvalDataset, self).__init__(eval_data, tokeniser, max_seq_len, num_derangements)
        self.sel_indices, self.sent_perms = dict(), dict()
        self._generate_data()

    def __getitem__(self, index: int):
        token_ids, token_masks = self.data[index]
        num_sents = token_ids.size(0)
        sel_indices, sent_perm = self.sel_indices[index], self.sent_perms[index]
        token_ids[sel_indices] = token_ids[sel_indices][sent_perm]
        token_masks[sel_indices] = token_masks[sel_indices][sent_perm]

        sent_perm = torch.tensor(sent_perm, dtype=torch.long)
        sel_indices = torch.tensor(sel_indices, dtype=torch.long)
        sent_position_id = torch.arange(num_sents).scatter_(0, sel_indices, value=num_sents)
        sent_type_id = torch.zeros(num_sents, dtype=torch.long).scatter_(0, sel_indices, value=1)
        return token_ids, token_masks, sent_position_id, sent_type_id, sent_perm

    def _generate_data(self):
        for index, (token_ids, token_masks) in tqdm.tqdm(enumerate(self.data), desc="Generating valset"):
            num_sents = token_ids.size(0)
            self.sel_indices[index], self.sent_perms[index] = self._generate_derangement(num_sents)
