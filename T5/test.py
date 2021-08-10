import argparse
import datasets
import jsonlines
import logging
import math
import nltk
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import shutil
import time
import torch
import wandb
from nlp import load_metric
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from data_set import StaticDataset

device = 'cuda:0' if cuda.is_available() else 'cpu'


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    with tqdm(loader, unit='batch') as tepoch:
        for _, data in enumerate(loader, 0):
            tepoch.set_description(f"Epoch {epoch}")
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, labels=y)
            loss = outputs.loss
            if _ % 10 == 0:
                wandb.log({"Training Loss": loss.item()})

            if _ % 500 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())


def validate(tokenizer, model, device, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            if _ % 100 == 0:
                print(f'Completed {_}')
            outputs = model(input_ids=ids, labels=y)
            loss = outputs.loss
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
    return avg_loss


def test(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    targets = []
    losses = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=250,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 100 == 0:
                print(f'Completed {_}')
            outputs = model(input_ids=ids, labels=y)
            loss = outputs.loss
            losses.append(loss)
            predictions.extend(preds)
            targets.extend(target)
        avg_loss = torch.stack(losses).mean()
    return targets, predictions, avg_loss


def main():
    # WandB – Initialize a new run
    wandb.init(project="T5")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config  # Initialize config
    config.TRAIN_BATCH_SIZE = 1  # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1  # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 10  # number of epochs to train (default: 10)
    config.LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
    config.SEED = 42  # random seed (default: 42)
    config.MAX_LEN = 1024
    config.SUMMARY_LEN = 250

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    model_name = 't5-small'
    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    training_set = StaticDataset('../data_generator/train_section_slides_t5.json', tokenizer, config.MAX_LEN,
                                 config.SUMMARY_LEN)
    val_set = StaticDataset('../data_generator/val_section_slides_t5.json', tokenizer, config.MAX_LEN,
                            config.SUMMARY_LEN)
    test_set = StaticDataset('../data_generator/tester.json', tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)

    phase = 'test'
    if phase == 'train':
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
        # in training phase we can distribute on multiple gpus
        device_map = {0: [0],
                      1: [2],
                      2: [4, 1],
                      3: [5, 3]}
        model.parallelize(device_map)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
        wandb.watch(model, log="all")
        loss = math.inf
        for epoch in range(config.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer)
            avg_loss = validate(tokenizer, model, device, val_loader)
            epoch_loss = avg_loss.item()
            wandb.log({"Val Loss": epoch_loss})
            print('val loss::', epoch_loss)
            if epoch_loss < loss:
                loss = epoch_loss
                model.save_pretrained('model_checkpoint/')
                # torch.save(model.state_dict, 'model_checkpoint/best.pt')
    elif phase == 'test':
        model = T5ForConditionalGeneration.from_pretrained('model_checkpoint/')
        # in test mode the model must be located on single gpu
        device_map = {0: [0, 1, 2, 3, 4, 5]}
        model.parallelize(device_map)
        wandb.watch(model, log="all")
        targets, predictions, avg_loss = test(tokenizer, model, device, test_loader)
        with open('t5_generated_summaries.txt', 'w') as f:
            for prediction in predictions:
                f.write(prediction + '\n')
        rouge = datasets.load_metric('rouge')
        rouge_dict = rouge.compute(predictions=predictions, references=targets)
        print('---- Rouge 1 Recall: {}, Rouge 2 Recall: {}, Rouge L Recall: {}, '.format(
            rouge_dict["rouge1"].mid.recall * 100, rouge_dict["rouge2"].mid.recall * 100,
            rouge_dict["rougeL"].mid.recall * 100))


if __name__ == '__main__':
    main()
