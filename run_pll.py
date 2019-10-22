import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from data import TextTrainDataset, TextEvalDataset, WikiText
from models import BertForPLL
from radam import RAdam
from train import train_pll, eval_pll


def parse_args():
    parser = argparse.ArgumentParser(description='PPL')
    parser.add_argument('--data_file',
                        type=str,
                        help='Path to the data folder',
                        required=True)
    parser.add_argument('--config_file',
                        type=str,
                        help='Path to the config file',
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

    wiki_text = WikiText(args.data_file)
    train_txt, val_txt, test_txt = wiki_text.splits()

    tokeniser = BertTokenizer.from_pretrained('bert-base-cased')
    with open(args.config_file, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    json_config["vocab_size_or_config_json_file"] = tokeniser.vocab_size
    args.ngram = json_config["max_position_embeddings"]
    bert_config = BertConfig(**json_config)

    model = BertForPLL(bert_config)
    model = torch.nn.DataParallel(model).to(device)
    optimiser = RAdam(model.parameters())

    if args.eval:
        dataloaders = {"test": DataLoader(TextEvalDataset(test_txt, args.ngram, tokeniser),
                                          batch_size=args.eval_batch_size,
                                          shuffle=False)}
        if args.resume:
            resume(model, args)

        test_loss, test_acc = eval_pll(device, model, dataloaders["test"], args)
        logger.info(f"Eval: Test Loss = {test_loss}, Test Acc = {test_acc}")
    else:
        dataloaders = {
            "train": DataLoader(TextTrainDataset(train_txt, args.ngram, tokeniser, args.poisson_rate),
                                batch_size=args.train_batch_size,
                                shuffle=False),
            "val": DataLoader(TextEvalDataset(val_txt, args.ngram, tokeniser),
                              batch_size=args.eval_batch_size,
                              shuffle=False),
            "test": DataLoader(TextEvalDataset(test_txt, args.ngram, tokeniser),
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
