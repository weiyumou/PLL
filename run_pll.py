import argparse

import torch
import torchtext
from torchtext.data.utils import get_tokenizer
import data
import numpy as np
import random


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
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root=args.data_dir)
    TEXT.build_vocab(train_txt)

    batch_size = 20
    eval_batch_size = 10
    train_data = data.TextDataset(train_txt, batch_size, TEXT)
    # val_data = data.TextDataset(val_txt, eval_batch_size, TEXT)
    # test_data = data.TextDataset(test_txt, eval_batch_size, TEXT)


if __name__ == '__main__':
    main()
