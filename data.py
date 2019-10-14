import random

import torch
import tqdm
from nltk.util import ngrams
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


class TextDataset(Dataset):
    def __init__(self, text_data, ngram, TEXT) -> None:
        super(TextDataset, self).__init__()
        all_ngrams = self.parse_ngrams(text_data, ngram)
        self.data = TEXT.numericalize(all_ngrams).permute(1, 0)  # torch.Size([4294, 9])

    def __len__(self) -> int:
        return self.data.size(0)

    @staticmethod
    def parse_ngrams(data, ngram):
        paras = " ".join(data.examples[0].text).split("<eos>")
        # nlp = spacy.load("en_core_web_sm")
        all_ngrams = []
        for i, para in tqdm.tqdm(enumerate(paras), desc="Parsing Paragraphs", total=len(paras)):
            para = para.strip()
            if para and not para.startswith("="):
                for item in ngrams(para.split(), ngram):
                    all_ngrams.append(item)

                # # Segment sentences
                # for sent in nlp(para).sents:
                #     # Generate ngrams
                #     for item in ngrams(sent.text.split(), ngram):
                #         all_ngrams.append(item)
        return all_ngrams


class TextTrainDataset(TextDataset):
    def __init__(self, text_data, ngram, TEXT, rate) -> None:
        super().__init__(text_data, ngram, TEXT)
        self.pdist = Poisson(rate=rate)
        self.ngram = ngram
        self.ds = calc_dn(ngram)

    def __getitem__(self, index: int):
        k = self.pdist.sample().int().item()
        perm = torch.tensor(k_permute(self.ngram, k, self.ds), dtype=torch.long)
        return self.data[index, perm], perm

    def set_poisson_rate(self, rate):
        self.pdist = Poisson(rate)


class TextEvalDataset(TextDataset):
    def __init__(self, text_data, ngram, TEXT) -> None:
        super().__init__(text_data, ngram, TEXT)
        ds = calc_dn(ngram)
        perms = []
        for i in range(len(self)):
            perms.append(k_permute(ngram, ngram, ds))
        self.perms = torch.tensor(perms, dtype=torch.long)

    def __getitem__(self, index: int):
        perm = self.perms[index]
        return self.data[index, perm], perm
