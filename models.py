import torch
import torch.nn as nn

from transformers import BertModel


class BertForPLL(nn.Module):

    def __init__(self, config):
        super(BertForPLL, self).__init__()
        self.bert = BertModel(config)
        self.pooler = nn.Linear(in_features=config.max_position_embeddings, out_features=1)
        self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.max_position_embeddings)
        nn.init.normal_(self.pooler.weight, 0, 0.01)
        nn.init.constant_(self.pooler.bias, 0)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        last_hidden_state, *_ = self.bert(x)  # (N, S, H)
        last_hidden_state = last_hidden_state.permute(0, 2, 1)  # (N, H, S)
        pooler_out = self.pooler(last_hidden_state)
        fc_out = self.fc(torch.relu(pooler_out.squeeze(-1)))
        return fc_out
