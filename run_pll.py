import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import RandomSampler, DistributedSampler
from transformers import BertTokenizer

import train
from data import WikiReader, GigawordReader, PLLTrainDataset, PLLEvalDataset
from models import HiBERTConfig, HiBERTWithAttn

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PPL")
    parser.add_argument("--deterministic",
                        help="Whether to set random seeds",
                        action="store_true")
    parser.add_argument("--data_root",
                        type=str,
                        help="Path to the data root",
                        required=True)
    parser.add_argument("--dtd_path",
                        type=str,
                        help="Path to the dtd file for Gigaword",
                        default=None)
    parser.add_argument("--invalid_file_path",
                        type=str,
                        help="Path to the list of invalid files for Gigaword",
                        default=None)
    parser.add_argument("--config",
                        type=str,
                        help="Path to a model config file",
                        default="config/1_4_BERT.json")
    parser.add_argument("--num_lines",
                        type=int,
                        help="Number of lines to read",
                        default=-1)
    parser.add_argument("--max_seq_len",
                        type=int,
                        help="Maximum sequence length",
                        default=64)
    parser.add_argument("--num_derangements",
                        type=int,
                        help="Number of derangements",
                        default=12)
    parser.add_argument("--sents_per_doc",
                        type=int,
                        help="Number of sentences per doc",
                        default=32)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="Number of epochs",
                        default=30)
    parser.add_argument("--lr", default=3e-4, type=float, help="The initial learning rate")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers used in data loading")
    parser.add_argument("--resume", default=None, type=str,
                        help="Path to a saved checkpoint")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup CUDA, GPU & distributed training
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        args.world_size = torch.distributed.get_world_size()

    if args.deterministic:
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(0)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, args.distributed, args.fp16)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.output_dir = os.path.join("models", time.ctime().replace(" ", "_").replace(":", "_"))
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.resume is not None:
        model = HiBERTWithAttn.from_pretrained(args.resume)
    else:
        with open(args.config, "r") as f:
            model_config = json.load(f)
        model_config.update({
            "vocab_size_or_config_json_file": tokeniser.vocab_size,
            "max_position_embeddings": (args.max_seq_len, args.sents_per_doc + 1),
            "type_vocab_size": (1, 2),
            "num_labels": (0, args.num_derangements)
        })
        model_config = HiBERTConfig(**model_config)
        model = HiBERTWithAttn(model_config)

    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    if args.dtd_path is None or args.invalid_file_path is None:
        data_reader = WikiReader(args.data_root, args.sents_per_doc, args.num_lines)
    else:
        data_reader = GigawordReader(args.data_root, args.dtd_path, args.invalid_file_path, args.sents_per_doc,
                                     args.num_lines)
    train_dataset = PLLTrainDataset(data_reader.train_set, tokeniser, args.max_seq_len, args.num_derangements)
    val_dataset = PLLEvalDataset(data_reader.val_set, tokeniser, args.max_seq_len, args.num_derangements)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    eval_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    dataloaders = {
        "train": DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True,
                            num_workers=args.num_workers),
        "val": DataLoader(val_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True,
                          num_workers=args.num_workers)
    }

    global_step, tr_loss = train.train_model(device, model, dataloaders, args, logger)
    logger.info(f"Global Steps = {global_step}, Average Loss = {tr_loss}")

    logger.info(f"Saving model checkpoint to {args.output_dir}")
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
