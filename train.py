import logging
import os

import torch
import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME

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


def train_model(device, model, dataloaders, tokeniser, num_epochs, num_segs, max_num_segs, model_dir):
    model = model.to(device)
    optimiser = RAdam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    curr_num_epochs = 0
    for epoch in range(num_epochs):
        running_loss, loss_total = 0, 0
        running_corrects, acc_total = 0, 0
        model.train()
        for input_ids, seg_ids, att_masks, perms in tqdm.tqdm(dataloaders["train"], desc="Training"):
            input_ids = input_ids.to(device)
            seg_ids = seg_ids.to(device)
            att_masks = att_masks.to(device)
            labels = torch.flatten(perms).to(device)

            outputs, *_ = model(input_ids=input_ids, attention_mask=att_masks, token_type_ids=seg_ids)
            outputs = outputs.reshape(-1, max_num_segs, max_num_segs)[:, :num_segs, :num_segs].reshape(-1, num_segs)

            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            preds = torch.argmax(outputs, dim=-1)
            running_corrects += torch.sum(preds == labels).item()
            acc_total += labels.numel()
            running_loss += loss.item() * labels.numel()
            loss_total += labels.numel()

        train_loss, train_acc = running_loss / loss_total, running_corrects / acc_total
        logger.info(f"Epoch {epoch}: # Segs = {num_segs}, Train Loss = {train_loss}, Train Acc = {train_acc}")

        # eval_loss, eval_acc = eval_model(device, model, dataloaders["val"], num_segs, max_num_segs)
        # logger.info(f"Epoch {epoch}: Val Loss = {eval_loss}, Val Acc = {eval_acc}")

        output_dir = os.path.join(model_dir, f"epoch_{epoch}-num_segs_{num_segs}")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model

        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokeniser.save_vocabulary(output_dir)

        curr_num_epochs += 1
        if curr_num_epochs == num_segs and num_segs < max_num_segs:
            curr_num_epochs = 0
            num_segs += 1
            dataloaders["train"].dataset.set_num_segs(num_segs)
            dataloaders["train"].dataset.set_poisson_rate(num_segs)


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
