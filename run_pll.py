import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import train
from data import WikiReader, WikiTrainDataset, WikiEvalDataset
from models import LSTMModel


def parse_args():
    parser = argparse.ArgumentParser(description='PPL')
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
                        action="store_true")
    parser.add_argument('--data_file',
                        type=str,
                        help='Path to the data file',
                        required=True)
    parser.add_argument('--num_lines',
                        type=int,
                        help='Number of lines to read',
                        default=-1)
    parser.add_argument('--max_seq_len',
                        type=int,
                        help='Maximum sequence length',
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
    parser.add_argument('--embed_dim',
                        type=int,
                        help='Size of the embeddings',
                        default=256)
    parser.add_argument('--hid_dim',
                        type=int,
                        help='Size of the hidden states',
                        default=128)
    parser.add_argument('--num_layers',
                        type=int,
                        help='Number of LSTM layers',
                        default=1)
    parser.add_argument('--bidir',
                        help='Whether the LSTM is bidirectional',
                        action="store_true")
    parser.add_argument('--init_pr',
                        type=int,
                        help='The initial poisson rate lambda',
                        default=2)
    parser.add_argument('--final_pr',
                        type=int,
                        help='The final poisson rate lambda',
                        default=32)
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

    wiki_reader = WikiReader(args.data_file, args.num_lines, args.max_seq_len, args.do_lower)
    wiki_reader.split()
    vocab_size = wiki_reader.vocab_size

    train_dataset = WikiTrainDataset(wiki_reader.train_set, args.max_seq_len, args.init_pr)
    val_dataset = WikiEvalDataset(wiki_reader.val_set, args.max_seq_len, args.final_pr)
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    }

    model = LSTMModel(vocab_size, args.embed_dim, args.hid_dim, args.max_seq_len, args.num_layers, args.bidir)

    train.train_model(device, model, dataloaders, args.num_epochs)


if __name__ == '__main__':
    main()
