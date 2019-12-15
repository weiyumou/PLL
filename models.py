import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class HiBERT(BertPreTrainedModel):
    def __init__(self, sent_enc_config, doc_enc_config):
        super(HiBERT, self).__init__(sent_enc_config)
        self.sent_enc = BertModel(sent_enc_config)
        self.doc_enc = BertModel(doc_enc_config)
        self.dropout = nn.Dropout(doc_enc_config.hidden_dropout_prob)
        self.classifier = nn.Linear(doc_enc_config.hidden_size, doc_enc_config.num_labels)

    def forward(self, token_ids, token_seg_ids, token_masks, n, s):
        sent_enc_out, *_ = self.sent_enc(input_ids=token_ids, attention_mask=token_masks, token_type_ids=token_seg_ids)
        sent_enc_out = torch.split(sent_enc_out[token_masks], torch.sum(token_masks, dim=-1).tolist())
        sent_enc_out = torch.stack([torch.mean(item, dim=0) for item in sent_enc_out], dim=0).reshape(n, s, -1)
        doc_enc_out, *_ = self.doc_enc(inputs_embeds=sent_enc_out)
        doc_enc_out = self.dropout(doc_enc_out)
        logits = self.classifier(doc_enc_out).reshape(n * s, -1)
        return logits
