import argparse
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
from data import WikiReader, WikiTrainDataset, WikiEvalDataset
from models import HiBERT, HiBERTConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PPL")
    parser.add_argument("--deterministic",
                        help="Whether to set random seeds",
                        action="store_true")
    parser.add_argument("--data_file",
                        type=str,
                        help="Path to the data file",
                        required=True)
    parser.add_argument("--num_lines",
                        type=int,
                        help="Number of lines to read",
                        default=-1)
    parser.add_argument("--max_seq_len",
                        type=int,
                        help="Maximum sequence length",
                        default=64)
    parser.add_argument("--num_paras",
                        type=int,
                        help="Number of paragraphs",
                        default=9)
    parser.add_argument("--sents_per_doc",
                        type=int,
                        help="Number of sentences per doc",
                        default=64)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="Number of epochs",
                        default=30)
    parser.add_argument("--sent_pr",
                        type=int,
                        help="The initial poisson rate lambda",
                        default=2)
    parser.add_argument("--learn_prd",
                        type=int,
                        help="Number of steps before providing harder examples",
                        default=1000)
    parser.add_argument("--eval",
                        help="Whether to evaluate the model",
                        action="store_true")
    parser.add_argument("--resume",
                        type=str,
                        help="Path to a saved checkpoint",
                        default=None)
    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        type=int,
                        help="Batch size per GPU/CPU for evaluation",
                        default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
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
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
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
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    tokeniser = BertTokenizer.from_pretrained("bert-base-cased")
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.output_dir = os.path.join("models", time.ctime().replace(" ", "_").replace(":", "_"))
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.resume is not None:
        model = HiBERT.from_pretrained(args.resume)
    else:
        model_config = {
            "vocab_size_or_config_json_file": tokeniser.vocab_size,
            "hidden_size": 192,  # 768
            "num_hidden_layers": 3,  # 12
            "num_attention_heads": 3,  # 12
            "intermediate_size": 768,  # 3072
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": (args.max_seq_len, args.sents_per_doc),
            "type_vocab_size": (1, args.num_paras),
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "num_labels": (0, args.num_paras)
        }
        model_config = HiBERTConfig(**model_config)
        model = HiBERT(model_config)

    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    wiki_reader = WikiReader(args.data_file, args.sents_per_doc, args.num_lines)
    train_dataset = WikiTrainDataset(wiki_reader.train_set, tokeniser, args.max_seq_len, args.num_paras, args.sent_pr)
    val_dataset = WikiEvalDataset(wiki_reader.val_set, tokeniser, args.max_seq_len, args.num_paras)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    eval_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    dataloaders = {
        "train": DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True,
                            num_workers=4 * args.n_gpu),
        "val": DataLoader(val_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True,
                          num_workers=4 * args.n_gpu)
    }

    global_step, tr_loss = train.train_model(device, model, dataloaders, args, logger)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
