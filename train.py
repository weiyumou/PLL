import os

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from RAdam.radam import RAdam


def calc_ranks(x):
    indices = torch.argsort(x, dim=-1, descending=True)
    ranks = torch.arange(indices.size(1), device=x.device).expand_as(indices)
    preds = torch.empty_like(indices).scatter_(1, indices, ranks)
    return preds


def train_model(device, model, dataloaders, args, logger):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_epochs = args.max_steps // (len(dataloaders["train"]) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataloaders["train"]) // args.gradient_accumulation_steps * args.num_epochs

    optimiser = RAdam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Apex
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimiser = amp.initialize(model, optimiser, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataloaders["train"].dataset))
    logger.info("  Num Epochs = %d", args.num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    curr_num_steps, global_step = 0, 0
    num_segs = args.num_segs
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    if args.resume is not None:
        checkpoint = torch.load(os.path.join(args.resume, "states.bin"))
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        curr_num_steps, global_step = checkpoint["curr_num_steps"], checkpoint["global_step"]
        num_segs = checkpoint["num_segs"]
    train_iterator = tqdm.trange(args.num_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        model.train()
        epoch_iterator = tqdm.tqdm(dataloaders["train"], desc="Iteration", disable=args.local_rank not in [-1, 0])
        running_corrects, acc_total = 0, 0
        for step, batch in enumerate(epoch_iterator):
            input_ids, seg_ids, att_masks, perms = batch
            input_ids = input_ids.to(device)
            seg_ids = seg_ids.to(device)
            att_masks = att_masks.to(device)
            labels = torch.flatten(perms).to(device)

            outputs, *_ = model(input_ids=input_ids, attention_mask=att_masks, token_type_ids=seg_ids)
            outputs = outputs.reshape(-1, args.max_num_segs, args.max_num_segs)[:, :num_segs, :num_segs].reshape(-1,
                                                                                                                 num_segs)

            loss = criterion(outputs, labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimiser) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()

            if args.local_rank in [-1, 0]:
                preds = torch.argmax(outputs, dim=-1)
                running_corrects += torch.sum(preds == labels).item()
                acc_total += labels.numel()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimiser), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimiser.step()
                model.zero_grad()
                global_step += 1
                curr_num_steps += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    train_acc = running_corrects / acc_total
                    loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar("Train_Acc", train_acc, global_step)
                    tb_writer.add_scalar("Train_Loss", loss, global_step)
                    tb_writer.add_scalar("Num_Segs", num_segs, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and curr_num_steps == num_segs * args.learn_prd and num_segs < args.max_num_segs:
                    curr_num_steps = 0
                    num_segs += 1
                    dataloaders["train"].dataset.set_num_segs(num_segs)
                    dataloaders["train"].dataset.set_poisson_rate(num_segs)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(
                        {"optimiser_state_dict": optimiser.state_dict(),
                         "global_step": global_step,
                         "num_segs": num_segs,
                         "curr_num_steps": curr_num_steps}, os.path.join(output_dir, "states.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def eval_model(device, model, dataloader, num_segs, max_num_segs):
    model = model.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss()

    running_loss, loss_total = 0, 0
    running_corrects, acc_total = 0, 0
    with torch.no_grad():
        for input_ids, seg_ids, att_masks, perms in tqdm.tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            seg_ids = seg_ids.to(device)
            att_masks = att_masks.to(device)
            labels = torch.flatten(perms).to(device)

            outputs, *_ = model(input_ids=input_ids, attention_mask=att_masks, token_type_ids=seg_ids)
            outputs = outputs.reshape(-1, max_num_segs, max_num_segs)[:, :num_segs, :num_segs].reshape(-1, num_segs)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=-1)
            running_corrects += torch.sum(preds == labels).item()
            acc_total += labels.numel()
            running_loss += loss.item() * labels.numel()
            loss_total += labels.numel()

    eval_loss, eval_acc = running_loss / loss_total, running_corrects / acc_total
    return eval_loss, eval_acc
