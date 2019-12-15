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


class WikiDataset(Dataset):
    def __init__(self, docs, tokeniser, max_seq_len, num_segs, sents_per_doc):
        super(WikiDataset, self).__init__()
        seq_len = max_seq_len - num_segs - 1
        self.data = []
        self.num_segs = num_segs
        for doc in docs:
            self.data.append([self._split_to_segs(tokeniser.tokenize(sent)[:seq_len]) for sent in doc])
        self.tokeniser = tokeniser
        self.max_seq_len = max_seq_len
        self.sents_per_doc = sents_per_doc
        self.token_ds = calc_dn(num_segs)
        self.sent_ds = calc_dn(sents_per_doc)

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
            tok_ids = self.tokeniser.convert_tokens_to_ids(seg + ["[SEP]"])
            token_ids.extend(tok_ids)
            seg_ids.extend([idx] * len(tok_ids))

        token_ids = torch.tensor(token_ids)
        seg_ids = torch.tensor(seg_ids)
        mask = torch.ones_like(token_ids, dtype=torch.long)

        token_ids = torch.nn.functional.pad(token_ids, [0, self.max_seq_len - token_ids.size(0)])
        seg_ids = torch.nn.functional.pad(seg_ids, [0, self.max_seq_len - seg_ids.size(0)])
        mask = torch.nn.functional.pad(mask, [0, self.max_seq_len - mask.size(0)])

        return token_ids, seg_ids, mask

    def __len__(self):
        return len(self.data)


class WikiTrainDataset(WikiDataset):
    def __init__(self, train_data, tokeniser, max_seq_len, num_segs, sents_per_doc, token_pr, sent_pr):
        super(WikiTrainDataset, self).__init__(train_data, tokeniser, max_seq_len, num_segs, sents_per_doc)
        self.token_pdist = Poisson(rate=token_pr)
        self.sent_pdist = Poisson(rate=sent_pr)

    def __getitem__(self, index: int):
        doc = []
        for sent_segs in self.data[index]:
            k = self.token_pdist.sample().int().item()
            perm = k_permute(self.num_segs, k, self.token_ds)
            doc.append(self._prep_inputs(sent_segs, perm))
        token_ids, token_seg_ids, token_masks = [torch.stack(item) for item in zip(*doc)]

        # sent_segs_id = torch.arange(self.sents_per_doc, dtype=torch.long)
        # sent_mask = torch.ones(self.sents_per_doc, dtype=torch.long)
        k = self.sent_pdist.sample().int().item()
        sent_perm = torch.tensor(k_permute(self.sents_per_doc, k, self.sent_ds))

        token_ids = token_ids[sent_perm]
        token_seg_ids = token_seg_ids[sent_perm]
        token_masks = token_masks[sent_perm]

        # num_pads = self.sents_per_doc - num_sents
        # token_ids = torch.nn.functional.pad(token_ids, [0, 0, 0, num_pads])
        # token_seg_ids = torch.nn.functional.pad(token_seg_ids, [0, 0, 0, num_pads])
        # token_masks = torch.nn.functional.pad(token_masks, [0, 0, 0, num_pads])
        # sent_segs_id = torch.nn.functional.pad(sent_segs_id, [0, num_pads])
        # sent_mask = torch.nn.functional.pad(sent_mask, [0, num_pads])
        # sent_perm = torch.nn.functional.pad(sent_perm, [0, num_pads])

        return token_ids, token_seg_ids, token_masks, sent_perm

    def set_poisson_rate(self, token_pr, sent_pr):
        self.token_pdist = Poisson(token_pr)
        self.sent_pdist = Poisson(sent_pr)


class WikiEvalDataset(WikiDataset):
    def __init__(self, eval_data, tokeniser, max_seq_len, num_segs, sents_per_doc):
        super(WikiEvalDataset, self).__init__(eval_data, tokeniser, max_seq_len, num_segs, sents_per_doc)
        self.token_pdist = Poisson(rate=num_segs)
        self.sent_pdist = Poisson(rate=16)
        self.docs, self.sent_perms = dict(), dict()
        self.generate_data()

    def __getitem__(self, index: int):
        doc = self.docs[index]
        token_ids, token_seg_ids, token_masks = [torch.stack(item) for item in zip(*doc)]

        # sent_segs_id = torch.arange(self.sents_per_doc, dtype=torch.long)
        # sent_mask = torch.ones(self.sents_per_doc, dtype=torch.long)
        sent_perm = self.sent_perms[index]

        token_ids = token_ids[sent_perm]
        token_seg_ids = token_seg_ids[sent_perm]
        token_masks = token_masks[sent_perm]

        return token_ids, token_seg_ids, token_masks, sent_perm

    def generate_data(self):
        for idx, doc in enumerate(self.data):
            self.docs[idx] = []
            for sent_segs in doc:
                k = self.token_pdist.sample().int().item()
                perm = k_permute(self.num_segs, k, self.token_ds)
                self.docs[idx].append(self._prep_inputs(sent_segs, perm))
            k = self.sent_pdist.sample().int().item()
            self.sent_perms[idx] = torch.tensor(k_permute(self.sents_per_doc, k, self.sent_ds))
