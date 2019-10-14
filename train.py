import os
import shutil

import torch
import torch.nn as nn
import tqdm
from torch.distributions import Geometric
from torch.utils.tensorboard import SummaryWriter


def train_pll(device, model, optimiser, dataloaders, args):
    criterion = nn.KLDivLoss(reduction="batchmean")
    gm_dist = Geometric(probs=torch.tensor([1 - 1e-5 ** (1 / args.ngram)]))
    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.num_epochs):
        running_loss, loss_total = 0, 0
        model.train()
        for inputs, perms in tqdm.tqdm(dataloaders["train"], desc="Training"):
            inputs = inputs.permute(1, 0).to(device)  # (S, N)
            labels = torch.exp(gm_dist.log_prob(perms.float())).to(device)  # (N, S)

            # Forward
            outputs = model(inputs)  # (N, S)
            outputs = torch.log_softmax(outputs.permute(1, 0), dim=-1)
            loss = criterion(outputs, labels)

            # Backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * labels.size(0)
            loss_total += labels.size(0)

        train_loss = running_loss / loss_total
        args.logger.info(f"Epoch {epoch}: Train Loss = {train_loss}")

        val_loss, val_acc = eval_pll(device, model, dataloaders["val"], args)
        args.logger.info(f"Epoch {epoch}: Val Loss = {val_loss}, Val Acc = {val_acc}")
        writer.add_scalars("Loss", {"train_loss": train_loss, "val_loss": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"val_acc": val_acc}, epoch)
        writer.add_scalar("Poisson_Rate", dataloaders["train"].dataset.pdist.rate, epoch)

        if (epoch + 1) % args.learn_prd == 0:
            args.poisson_rate += 1
            dataloaders["train"].dataset.set_poisson_rate(args.poisson_rate)

        # save checkpoints
        is_best = val_acc > args.best_acc
        args.best_acc = max(val_acc, args.best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': args.best_acc,
            'optimiser': optimiser.state_dict(),
            "poisson_rate": args.poisson_rate
        }, is_best, args.model_dir)


def eval_pll(device, model, dataloader, args):
    model.eval()
    criterion = nn.KLDivLoss(reduction="batchmean")
    gm_dist = Geometric(probs=torch.tensor([1 - 1e-5 ** (1 / args.ngram)]))
    running_loss, loss_total = 0, 0
    running_corrects, accuracy_total = 0, 0

    for inputs, perms in tqdm.tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.permute(1, 0).to(device)  # (S, N)
        labels = torch.exp(gm_dist.log_prob(perms.float())).to(device)  # (N, S)
        with torch.no_grad():
            outputs = model(inputs)  # (N, S)
            outputs = torch.log_softmax(outputs.permute(1, 0), dim=-1)
            loss = criterion(outputs, labels)

        indices = torch.argsort(outputs, dim=-1, descending=True)
        ranks = torch.arange(args.ngram, device=device).expand_as(indices)
        preds = torch.empty_like(indices).scatter_(1, indices, ranks).cpu()
        running_corrects += torch.sum(preds == perms).item()
        accuracy_total += perms.numel()
        running_loss += loss.item() * labels.size(0)
        loss_total += labels.size(0)

    return running_loss / loss_total, running_corrects / accuracy_total


def save_checkpoint(state, is_best, model_dir):
    filename = os.path.join(model_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, "model_best.pth.tar"))



