import random

import torch
import tqdm
from torch.distributions import Poisson
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


class WikiSentDataset(Dataset):

    def __init__(self, docs, tokeniser, max_seq_len) -> None:
        super(WikiSentDataset, self).__init__()
        self.data = docs
        self.tokeniser = tokeniser
        # for doc in docs:
        #     self.data.append([tokeniser.encode(sent, max_length=max_seq_len) for sent in doc])
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
    def __init__(self, docs, tokeniser, max_seq_len, num_paras):
        super(WikiDataset, self).__init__()
        self.data = WikiSentDataset(docs, tokeniser, max_seq_len)
        self.num_paras = num_paras
        self.ds = calc_dn(num_paras)

    def _calc_para_lens(self, num_sents):
        para_lens = [num_sents // self.num_paras] * self.num_paras
        rand_inds = random.sample(range(self.num_paras), num_sents % self.num_paras)
        for idx in rand_inds:
            para_lens[idx] += 1
        return para_lens

    def __len__(self):
        return len(self.data)


class WikiTrainDataset(WikiDataset):
    def __init__(self, train_data, tokeniser, max_seq_len, num_paras, pr):
        super(WikiTrainDataset, self).__init__(train_data, tokeniser, max_seq_len, num_paras)
        self.pdist = Poisson(rate=pr)

    def __getitem__(self, index: int):
        token_ids, token_masks = self.data[index]
        num_sents = token_ids.size(0)
        para_lens = self._calc_para_lens(num_sents)
        para_ids = torch.repeat_interleave(torch.arange(self.num_paras), torch.tensor(para_lens))

        # Permute paragraphs
        k = self.pdist.sample().int().item()
        sent_perm = k_permute(self.num_paras, k, self.ds)
        token_ids = torch.split(token_ids, para_lens, dim=0)
        token_ids = torch.cat([token_ids[i] for i in sent_perm], dim=0)
        token_masks = torch.split(token_masks, para_lens, dim=0)
        token_masks = torch.cat([token_masks[i] for i in sent_perm], dim=0)
        sent_perm = torch.tensor(sent_perm, dtype=torch.long)
        return token_ids, token_masks, para_ids, sent_perm

    def set_poisson_rate(self, pr):
        self.pdist = Poisson(pr)


class WikiEvalDataset(WikiDataset):
    def __init__(self, eval_data, tokeniser, max_seq_len, num_paras):
        super(WikiEvalDataset, self).__init__(eval_data, tokeniser, max_seq_len, num_paras)
        self.pdist = Poisson(rate=num_paras * 2)
        self.para_lens, self.sent_perms = dict(), dict()
        self.generate_data()

    def __getitem__(self, index: int):
        token_ids, token_masks = self.data[index]
        para_lens = self.para_lens[index]
        para_ids = torch.repeat_interleave(torch.arange(self.num_paras), torch.tensor(para_lens))

        # Permute paragraphs
        sent_perm = self.sent_perms[index]
        token_ids = torch.split(token_ids, para_lens, dim=0)
        token_ids = torch.cat([token_ids[i] for i in sent_perm], dim=0)
        token_masks = torch.split(token_masks, para_lens, dim=0)
        token_masks = torch.cat([token_masks[i] for i in sent_perm], dim=0)
        sent_perm = torch.tensor(sent_perm, dtype=torch.long)
        return token_ids, token_masks, para_ids, sent_perm

    def generate_data(self):
        for idx, (token_ids, token_masks) in enumerate(self.data):
            num_sents = token_ids.size(0)
            self.para_lens[idx] = self._calc_para_lens(num_sents)
            k = self.pdist.sample().int().item()
            self.sent_perms[idx] = k_permute(self.num_paras, k, self.ds)
