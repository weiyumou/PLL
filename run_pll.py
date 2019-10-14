import argparse
import logging
import os
import random
import time
import json

import numpy as np
import torch
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from data import TextTrainDataset, TextEvalDataset
from models import TransformerModel
from train import train_pll, eval_pll


def parse_args():
    parser = argparse.ArgumentParser(description='PPL')
    parser.add_argument('--data_dir',
                        type=str,
                        help='Path to the data folder',
                        required=True)
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
    parser.add_argument('--eval',
                        help='Whether to evaluate the model',
                        action="store_true")
    parser.add_argument('--resume',
                        type=str,
                        help='Path to a saved checkpoint',
                        default="")
    args = parser.parse_args()
    return args


def resume(model, args, optimiser=None):
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
        args.start_epoch = checkpoint['epoch']
        args.best_acc = checkpoint['best_acc']
        args.poisson_rate = checkpoint["poisson_rate"]
        model.load_state_dict(checkpoint['state_dict'])
        if optimiser is not None:
            optimiser.load_state_dict(checkpoint['optimiser'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")


def main():
    args = parse_args()
    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu = 0

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=False)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root=args.data_dir)
    TEXT.build_vocab(train_txt)

    model = TransformerModel(len(TEXT.vocab.stoi), args.em_size,
                             args.num_heads, args.hid_size, args.num_layers).to(device)
    model = torch.nn.DataParallel(model, dim=1)
    optimiser = optim.Adam(model.parameters())

    if args.eval:
        dataloaders = {"test": DataLoader(TextEvalDataset(test_txt, args.ngram, TEXT),
                                          batch_size=args.eval_batch_size,
                                          shuffle=False)}
        if args.resume:
            resume(model, args)

        test_loss, test_acc = eval_pll(device, model, dataloaders["test"], args)
        logger.info(f"Eval: Test Loss = {test_loss}, Test Acc = {test_acc}")
    else:
        dataloaders = {
            "train": DataLoader(TextTrainDataset(train_txt, args.ngram, TEXT, args.poisson_rate),
                                batch_size=args.train_batch_size,
                                shuffle=True),
            "val": DataLoader(TextEvalDataset(val_txt, args.ngram, TEXT),
                              batch_size=args.eval_batch_size,
                              shuffle=False),
            "test": DataLoader(TextEvalDataset(test_txt, args.ngram, TEXT),
                               batch_size=args.eval_batch_size,
                               shuffle=False)
        }
        args.start_epoch = 0
        args.best_acc = 1 / args.ngram
        if args.resume:
            resume(model, args, optimiser)

        # Create folder for the current model and save args
        model_dir = time.ctime().replace(" ", "_").replace(":", "_")
        args.model_dir = os.path.join("models", model_dir)
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        args.logger = logger
        train_pll(device, model, optimiser, dataloaders, args)


if __name__ == '__main__':
    main()
