import logging

import numpy as np
import torch
import tqdm

from RAdam.radam import RAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def calc_ranks(x):
    indices = torch.argsort(x, dim=-1, descending=True)
    ranks = torch.arange(indices.size(1), device=x.device).expand_as(indices)
    preds = torch.empty_like(indices).scatter_(1, indices, ranks)
    return preds


def train_model(device, model, dataloaders, num_epochs):
    model = model.to(device)
    optimiser = RAdam(model.parameters())
    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(num_epochs):
        running_loss, loss_total = 0, 0
        running_corrects, acc_total = 0, 0
        model.train()
        for inputs, labels, seq_lens, masks in tqdm.tqdm(dataloaders["train"], desc="Training"):
            inputs = inputs.permute([1, 0]).to(device)
            labels = labels.to(device)
            outputs = model(inputs, seq_lens)
            outputs[~masks] = -np.inf
            outputs = torch.log_softmax(outputs, dim=-1)

            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            preds, perms = calc_ranks(outputs), calc_ranks(labels)
            running_corrects += torch.sum(preds[masks] == perms[masks]).item()
            acc_total += torch.sum(masks).item()
            running_loss += loss.item() * labels.size(0)
            loss_total += labels.size(0)

        train_loss, train_acc = running_loss / loss_total, running_corrects / acc_total
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss}, Train Acc = {train_acc}")

        eval_loss, eval_acc = eval_model(device, model, dataloaders["val"])
        logger.info(f"Epoch {epoch}: Val Loss = {eval_loss}, Val Acc = {eval_acc}")


def eval_model(device, model, dataloader):
    model = model.to(device).eval()
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    running_loss, loss_total = 0, 0
    running_corrects, total = 0, 0
    with torch.no_grad():
        for inputs, labels, seq_lens, masks in tqdm.tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.permute([1, 0]).to(device)
            labels = labels.to(device)
            outputs = model(inputs, seq_lens)
            outputs[~masks] = -np.inf
            outputs = torch.log_softmax(outputs, dim=-1)

            loss = criterion(outputs, labels)
            preds, perms = calc_ranks(outputs), calc_ranks(labels)
            running_corrects += torch.sum(preds[masks] == perms[masks]).item()
            total += torch.sum(masks).item()
            running_loss += loss.item() * labels.size(0)
            loss_total += labels.size(0)

    eval_loss, eval_acc = running_loss / loss_total, running_corrects / total
    return eval_loss, eval_acc
