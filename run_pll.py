import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import train
from data import WikiReader, WikiTrainDataset


def parse_args():
    parser = argparse.ArgumentParser(description='PPL')
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
                        action="store_true")
    parser.add_argument('--data_file',
                        type=str,
                        help='Path to the data file',
                        required=True)
    parser.add_argument('--model_dir',
                        type=str,
                        help='Path to save models',
                        default="models")
    parser.add_argument('--num_lines',
                        type=int,
                        help='Number of lines to read',
                        default=-1)
    parser.add_argument('--max_seq_len',
                        type=int,
                        help='Maximum sequence length',
                        default=64)
    parser.add_argument('--num_segs',
                        type=int,
                        help='Number of segments',
                        default=-1)
    parser.add_argument('--max_num_segs',
                        type=int,
                        help='Maximum number of segments',
                        default=64)
    parser.add_argument('--do_lower',
                        help='Whether to use lower case letters',
                        action="store_true")
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs',
                        default=30)
    parser.add_argument('--train_batch_size',
                        type=int,
                        help='Train batch size for SL',
                        default=128)
    parser.add_argument('--eval_batch_size',
                        type=int,
                        help='Val batch size for SL',
                        default=128)
    parser.add_argument('--init_pr',
                        type=int,
                        help='The initial poisson rate lambda',
                        default=2)
    parser.add_argument('--learn_prd',
                        type=int,
                        help='Number of epochs before providing harder examples',
                        default=10)
    parser.add_argument('--eval',
                        help='Whether to evaluate the model',
                        action="store_true")
    parser.add_argument('--resume',
                        type=str,
                        help='Path to a saved checkpoint',
                        default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(args.model_dir, time.ctime().replace(" ", "_").replace(":", "_"))
    os.makedirs(model_dir, exist_ok=True)

    tokeniser = BertTokenizer.from_pretrained("bert-base-cased")
    config_json = {
        "vocab_size": tokeniser.vocab_size,
        "hidden_size": 192,  # 768
        "num_hidden_layers": 3,  # 12
        "num_attention_heads": 3,  # 12
        "intermediate_size": 768,  # 3072
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": args.max_seq_len,
        "type_vocab_size": args.max_num_segs,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "num_labels": args.max_num_segs ** 2
    }
    bert_config = BertConfig(**config_json)

    wiki_reader = WikiReader(args.data_file, args.num_lines)
    # wiki_reader.split(train_perct=0.9)

    train_dataset = WikiTrainDataset(wiki_reader.sentences, tokeniser,
                                     args.max_seq_len, args.init_pr,
                                     args.num_segs, args.max_num_segs)
    # val_dataset = WikiEvalDataset(wiki_reader.val_set, tokeniser, args.max_seq_len, args.max_num_segs)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        # "val": DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    }

    model = BertForSequenceClassification(bert_config)
    train.train_model(device, model, dataloaders,
                      tokeniser, args.num_epochs,
                      args.num_segs, args.max_num_segs, model_dir)


if __name__ == '__main__':
    main()
