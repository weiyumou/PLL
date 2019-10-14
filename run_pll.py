import argparse
import random

import numpy as np
import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from data import TextDataset
from models import TransformerModel
from train import train_pll


def parse_args():
    parser = argparse.ArgumentParser(description='PPL')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the data folder',
                        required=True)
    parser.add_argument('--model_dir',
                        type=str,
                        help='Path to the saved models',
                        default="models")
    parser.add_argument('--deterministic',
                        help='Whether to set random seeds',
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
    parser.add_argument('--em_size',
                        type=int,
                        help='Size of the embeddings',
                        default=256)
    parser.add_argument('--num_heads',
                        type=int,
                        help='Number of heads in the MultiheadAttention',
                        default=4)
    parser.add_argument('--hid_size',
                        type=int,
                        help='Size of the hidden states',
                        default=128)
    parser.add_argument('--num_layers',
                        type=int,
                        help='Number of TransformerEncoders',
                        default=4)
    parser.add_argument('--ngram',
                        type=int,
                        help='Number of words in a segment',
                        default=9)
    parser.add_argument('--learn_prd',
                        type=int,
                        help='Number of epochs before providing harder examples',
                        default=10)
    parser.add_argument('--poisson_rate',
                        type=int,
                        help='The initial poisson rate lambda',
                        default=2)
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

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=False)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root=args.data_dir)
    TEXT.build_vocab(train_txt)

    dataloaders = {
        "train": DataLoader(TextDataset(train_txt, args.ngram, TEXT),
                            batch_size=args.train_batch_size,
                            shuffle=False),
        "val": DataLoader(TextDataset(val_txt, args.ngram, TEXT),
                          batch_size=args.eval_batch_size,
                          shuffle=False),
        "test": DataLoader(TextDataset(test_txt, args.ngram, TEXT),
                           batch_size=args.eval_batch_size,
                           shuffle=False)
    }

    model = TransformerModel(len(TEXT.vocab.stoi), args.em_size,
                             args.num_heads, args.hid_size, args.num_layers)
    train_pll(device, model, dataloaders)


if __name__ == '__main__':
    main()
