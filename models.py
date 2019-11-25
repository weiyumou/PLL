import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hid_dim, max_seq_len, num_layers, bidir):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hid_dim, num_layers, bidirectional=bidir)
        # self.pooler = nn.Linear(in_features=max_seq_len, out_features=1)
        self.fc = nn.Linear(hid_dim * (bidir + 1), max_seq_len)
        self.num_layers = num_layers

    def forward(self, x, seq_lens):
        embed_out = nn.utils.rnn.pack_padded_sequence(self.embed(x), seq_lens, enforce_sorted=False)
        # outputs, *_ = self.encoder(embed_out)
        # outputs, lens = nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = outputs.permute([1, 2, 0])  # (N, 2H, S)
        _, (h_n, _) = self.encoder(embed_out)
        h_n = h_n.reshape(self.num_layers, -1, h_n.size(1), h_n.size(2))
        h_n = torch.flatten(h_n[-1].permute([1, 0, 2]), start_dim=1)
        # pooler_out = self.pooler(outputs).squeeze(dim=-1)
        fc_out = self.fc(h_n)
        return fc_out
