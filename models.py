import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig


class HiBERTConfig(BertConfig):

    def __init__(self, vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2,
                 initializer_range=0.02, layer_norm_eps=1e-12, **kwargs):
        super().__init__(vocab_size_or_config_json_file, hidden_size, num_hidden_layers, num_attention_heads,
                         intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob,
                         max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps)

        self.num_labels = kwargs.pop("num_labels", 2)
        self.sent_enc_config, self.doc_enc_config = None, None
        self.create_bert_config()

    def create_bert_config(self):
        dict_cpy = copy.deepcopy(self.__dict__)
        dict_cpy["vocab_size"] = (dict_cpy["vocab_size"], 1)
        del dict_cpy["sent_enc_config"], dict_cpy["doc_enc_config"]
        for key in dict_cpy:
            if not (isinstance(dict_cpy[key], list) or isinstance(dict_cpy[key], tuple)):
                dict_cpy[key] = (dict_cpy[key], dict_cpy[key])
        values = list(zip(*dict_cpy.values()))
        sent_enc_config = dict(zip(dict_cpy.keys(), values[0]))
        sent_enc_config["vocab_size_or_config_json_file"] = sent_enc_config.pop("vocab_size")
        doc_enc_config = dict(zip(dict_cpy.keys(), values[1]))
        doc_enc_config["vocab_size_or_config_json_file"] = doc_enc_config.pop("vocab_size")
        self.sent_enc_config = BertConfig(**sent_enc_config)
        self.doc_enc_config = BertConfig(**doc_enc_config)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        del output["sent_enc_config"], output["doc_enc_config"]
        return output

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = super().from_dict(json_object)
        config.create_bert_config()
        return config


class HiBERT(BertPreTrainedModel):
    config_class = HiBERTConfig

    def __init__(self, model_config):
        super(HiBERT, self).__init__(model_config.sent_enc_config)
        self.config = model_config
        self.sent_enc = BertModel(model_config.sent_enc_config)
        self.doc_enc = BertModel(model_config.doc_enc_config)
        self.classifier = nn.Linear(model_config.doc_enc_config.hidden_size, model_config.doc_enc_config.num_labels)
        nn.init.normal_(self.classifier.weight, std=model_config.doc_enc_config.initializer_range)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, token_ids, token_masks, sent_position_ids, sent_type_ids):
        n, s = sent_position_ids.size()
        sent_enc_out, *_ = self.sent_enc(input_ids=token_ids, attention_mask=token_masks)
        sent_enc_out = torch.mean(sent_enc_out, dim=1).reshape(n, s, -1)
        doc_enc_out, *_ = self.doc_enc(inputs_embeds=sent_enc_out,
                                       position_ids=sent_position_ids,
                                       token_type_ids=sent_type_ids)
        doc_enc_out = doc_enc_out[sent_type_ids.bool()]
        logits = self.classifier(doc_enc_out)
        return logits


class HiBERTWithAttn(HiBERT):

    def __init__(self, model_config):
        super(HiBERTWithAttn, self).__init__(model_config)
        self.attn_pool = AttentionPool(self.doc_enc.config.hidden_size,
                                       self.doc_enc.config.hidden_size // 2,
                                       self.doc_enc.config.initializer_range)

    def forward(self, token_ids, token_masks, sent_position_ids, sent_type_ids):
        n, s = sent_position_ids.size()
        sent_enc_out, *_ = self.sent_enc(input_ids=token_ids, attention_mask=token_masks)
        sent_enc_out = self.attn_pool(sent_enc_out).reshape(n, s, -1)
        doc_enc_out, *_ = self.doc_enc(inputs_embeds=sent_enc_out,
                                       position_ids=sent_position_ids,
                                       token_type_ids=sent_type_ids)
        doc_enc_out = doc_enc_out[sent_type_ids.bool()]
        logits = self.classifier(doc_enc_out)
        return logits


class AttentionPool(nn.Module):
    def __init__(self, hidden_size, attn_hidden_size, initializer_range) -> None:
        super(AttentionPool, self).__init__()
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=attn_hidden_size, bias=False)
        self.fc2 = nn.Linear(in_features=attn_hidden_size, out_features=1, bias=False)
        nn.init.normal_(self.fc1.weight, std=initializer_range)
        nn.init.normal_(self.fc2.weight, std=initializer_range)

    def forward(self, x):
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(torch.tanh(fc1_out)).permute([0, 2, 1])
        attn_weights = torch.softmax(fc2_out, dim=-1)
        return torch.bmm(attn_weights, x)
