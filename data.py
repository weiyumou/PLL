import itertools
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
    def __init__(self, data_file, line_per_doc, num_lines=-1):
        self.docs, seq = [], []
        curr_num_lines = 0
        with open(data_file, "r") as file_hdl:
            for line in tqdm.tqdm(file_hdl, desc="Reading File"):
                line = line.rstrip("\n")
                if not line:
                    continue
                if is_doc_start(line) or curr_num_lines == line_per_doc:
                    if seq:
                        self.docs.append(seq)
                        seq = []
                        curr_num_lines = 0
                    continue
                seq.append(line)
                curr_num_lines += 1
                num_lines -= 1
                if num_lines == 0:
                    break

    def split(self, train_perct=0.9):
        train_idx = int(train_perct * len(self.docs))
        self.train_set = self.docs[:train_idx]
        self.val_set = self.docs[train_idx:]
        self.docs = None


class WikiDataset(Dataset):
    def __init__(self, docs, tokeniser, max_seq_len, num_segs, sents_per_doc):
        super(WikiDataset, self).__init__()
        seq_len = max_seq_len - num_segs - 1
        self.data = []
        for doc in docs:
            self.data.append([tokeniser.tokenize(sent)[:seq_len] for sent in doc])
        self.tokeniser = tokeniser
        self.max_seq_len = max_seq_len
        self.num_segs = num_segs
        self.ds = calc_dn(num_segs)
        self.sents_per_doc = sents_per_doc

    def _split_to_segs(self, sentence):
        seq_len = len(sentence)
        seg_lens = [seq_len // self.num_segs] * self.num_segs
        rand_inds = random.sample(range(self.num_segs), seq_len % self.num_segs)
        for idx in rand_inds:
            seg_lens[idx] += 1
        sent_segs, last_idx = [], 0
        for idx in itertools.accumulate(seg_lens):
            sent_segs.append(sentence[last_idx: idx])
            last_idx = idx
        return sent_segs

    def _prep_inputs(self, sent_segs, perm):
        sent_segs = [sent_segs[i] for i in perm]
        sent_segs[0] = ["[CLS]"] + sent_segs[0]

        token_ids, seg_ids = [], []
        for idx, seg in enumerate(sent_segs):
            tok_ids = self.tokeniser.encode(" ".join(seg + ["[SEP]"]), add_special_tokens=False)
            token_ids.extend(tok_ids)
            seg_ids.extend([idx] * len(tok_ids))

        token_ids = torch.tensor(token_ids)
        seg_ids = torch.tensor(seg_ids)
        mask = torch.ones_like(token_ids, dtype=torch.long)

        token_ids = torch.nn.functional.pad(token_ids, [0, self.max_seq_len - token_ids.size(0)])
        seg_ids = torch.nn.functional.pad(seg_ids, [0, self.max_seq_len - seg_ids.size(0)])
        mask = torch.nn.functional.pad(mask, [0, self.max_seq_len - mask.size(0)])
        perm = torch.tensor(perm)

        return token_ids, seg_ids, mask, perm

    def __len__(self):
        return len(self.data)


class WikiTrainDataset(WikiDataset):
    def __init__(self, train_data, tokeniser, max_seq_len, num_segs, sents_per_doc, rate):
        super(WikiTrainDataset, self).__init__(train_data, tokeniser, max_seq_len, num_segs, sents_per_doc)
        self.pdist = Poisson(rate=rate)

    def __getitem__(self, index: int):
        doc = []
        for sentence in self.data[index]:
            sent_segs = self._split_to_segs(sentence)
            k = self.pdist.sample().int().item()
            perm = k_permute(self.num_segs, k, self.ds)
            doc.append(self._prep_inputs(sent_segs, perm))
        sent_token_ids, sent_seg_ids, sent_masks, sent_perms = [torch.stack(item) for item in zip(*doc)]
        doc_mask = torch.ones(sent_token_ids.size(0), dtype=torch.long)
        num_pads = self.sents_per_doc - sent_token_ids.size(0)
        sent_token_ids = torch.cat(
            [sent_token_ids, torch.zeros(num_pads, sent_token_ids.size(-1), dtype=torch.long)], dim=0)
        sent_seg_ids = torch.cat(
            [sent_seg_ids, torch.zeros(num_pads, sent_seg_ids.size(-1), dtype=torch.long)], dim=0)
        sent_masks = torch.cat(
            [sent_masks, torch.zeros(num_pads, sent_masks.size(-1), dtype=torch.long)], dim=0)
        sent_perms = torch.cat(
            [sent_token_ids, torch.zeros(num_pads, sent_perms.size(-1), dtype=torch.long)], dim=0)
        return sent_token_ids, sent_seg_ids, sent_masks, sent_perms

    def set_poisson_rate(self, rate):
        self.pdist = Poisson(rate)


class WikiEvalDataset(WikiDataset):
    def __init__(self, eval_data, tokeniser, max_seq_len, num_segs):
        super(WikiEvalDataset, self).__init__(eval_data, tokeniser, max_seq_len, num_segs)
        self.pdist = Poisson(rate=num_segs * 2)
        self.sent_segs, self.perms = dict(), dict()
        self.generate_data()

    def __getitem__(self, index: int):
        sent_segs, perm = self.sent_segs[index], self.perms[index]
        return self._prep_inputs(sent_segs, perm)

    def generate_data(self):
        for idx, sentence in enumerate(self.data):
            self.sent_segs[idx] = self._split_to_segs(sentence)
            k = self.pdist.sample().int().item()
            self.perms[idx] = k_permute(self.num_segs, k, self.ds)
