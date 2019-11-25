import random

import torch
import tqdm
from torch.distributions import Geometric
from torch.distributions import Poisson
from torch.utils.data import Dataset
from transformers import BertTokenizer


class DerangementGen:
    def __init__(self, max_seq_len):
        self.ds = self.calc_dn(max_seq_len)

    def sample(self, n, k):
        return self.k_permute(n, k, self.ds)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        perm = DerangementGen.random_derangement(k, ds)
        new_indices = indices.copy()
        for i, p in enumerate(perm):
            new_indices[sel_indices[i]] = indices[sel_indices[p]]
        return new_indices


class WikiReader(object):
    def __init__(self, data_file, num_lines=-1, max_seq_len=64, do_lower=False):
        self.sentences = []
        tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = tokeniser.vocab_size
        with open(data_file, "r") as file_hdl:
            for line in tqdm.tqdm(file_hdl, desc="Reading File"):
                line = line.strip("\n\r")
                if not line or line.startswith("["):
                    continue
                tokens = tokeniser.encode(line)[:max_seq_len]
                self.sentences.append(tokens)
                num_lines -= 1
                if num_lines == 0:
                    break

    def split(self, train_perct=0.7):
        train_idx = int(train_perct * len(self.sentences))
        val_idx = train_idx + int((1 - train_perct) / 2 * len(self.sentences))
        self.train_set = self.sentences[:train_idx]
        self.val_set = self.sentences[train_idx: val_idx]
        self.test_set = self.sentences[val_idx:]
        self.sentences = None


class WikiDataset(Dataset):
    def __init__(self, dataset):
        super(WikiDataset, self).__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)


class WikiTrainDataset(WikiDataset):
    def __init__(self, train_data, max_seq_len, init_rate):
        super(WikiTrainDataset, self).__init__(train_data)
        self.pdist = Poisson(rate=init_rate)
        self.de_gen = DerangementGen(max_seq_len)
        self.max_seq_len = max_seq_len

    def __getitem__(self, index: int):
        sentence = self.data[index]
        seq_len = len(sentence)

        k = self.pdist.sample().int().item()
        perm = self.de_gen.sample(seq_len, k)
        sentence = torch.tensor([sentence[i] for i in perm])

        gm_dist = Geometric(probs=torch.tensor([1 - 1e-5 ** (1 / seq_len)]))
        label = torch.exp(gm_dist.log_prob(torch.tensor(perm, dtype=float))).float()

        sentence = torch.cat([sentence, torch.zeros(self.max_seq_len - seq_len, dtype=sentence.dtype)])
        label = torch.cat([label, torch.zeros(self.max_seq_len - seq_len, dtype=label.dtype)])
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = 1
        return sentence, label, seq_len, mask

    def set_poisson_rate(self, rate):
        self.pdist = Poisson(rate)


class WikiEvalDataset(WikiDataset):
    def __init__(self, eval_data, max_seq_len, init_rate):
        super(WikiEvalDataset, self).__init__(eval_data)
        self.pdist = Poisson(rate=init_rate)
        self.de_gen = DerangementGen(max_seq_len)
        self.max_seq_len = max_seq_len
        self.perms = dict()
        for idx, sentence in enumerate(self.data):
            k = self.pdist.sample().int().item()
            self.perms[idx] = self.de_gen.sample(len(sentence), k)

    def __getitem__(self, index: int):
        sentence = self.data[index]
        seq_len = len(sentence)
        sentence = torch.tensor([sentence[i] for i in self.perms[index]])

        gm_dist = Geometric(probs=torch.tensor([1 - 1e-5 ** (1 / seq_len)]))
        label = torch.exp(gm_dist.log_prob(torch.tensor(self.perms[index], dtype=float))).float()

        sentence = torch.cat([sentence, torch.zeros(self.max_seq_len - seq_len, dtype=sentence.dtype)])
        label = torch.cat([label, torch.zeros(self.max_seq_len - seq_len, dtype=label.dtype)])
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = 1
        return sentence, label, seq_len, mask
