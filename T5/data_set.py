import json
import jsonlines
import pytorch_lightning as pl
import torch
from itertools import cycle
from torch.utils.data import DataLoader, IterableDataset, Dataset
from transformers import T5Tokenizer


class CustomDataset(IterableDataset):

    def __init__(self, path, tokenizer, source_len, summary_len):
        self.tokenizer = tokenizer
        self.path = path
        self.source_len = source_len
        self.summary_len = summary_len

    def __len__(self):
        return len(open(self.path).readlines())

    def parse_file(self):
        with jsonlines.open(self.path) as reader:
            for line in reader:
                source = self.tokenizer(line['text'], max_length=self.source_len, pad_to_max_length=True,
                                        truncation=True,
                                        return_tensors='pt')
                target = self.tokenizer(line['summary'], max_length=self.summary_len, pad_to_max_length=True,
                                        truncation=True,
                                        return_tensors='pt')

                source_ids = source.input_ids.squeeze()
                source_mask = source.attention_mask.squeeze()
                target_ids = target.input_ids.squeeze()
                target_mask = target.attention_mask.squeeze()
                yield {
                    'source_ids': source_ids.to(dtype=torch.long),
                    'source_mask': source_mask.to(dtype=torch.long),
                    'target_ids': target_ids.to(dtype=torch.long),
                    'target_mask': target_mask.to(dtype=torch.long)
                }

    def __iter__(self):
        return cycle(self.parse_file())


class StaticDataset(Dataset):

    def __init__(self, path, tokenizer, source_len, summary_len):
        self.tokenizer = tokenizer
        self.path = path
        self.source_len = source_len
        self.summary_len = summary_len
        self.data = open(self.path).readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = json.loads(self.data[index])
        source = self.tokenizer(line['text'], max_length=self.source_len, pad_to_max_length=True, truncation=True,
                                return_tensors='pt')
        target = self.tokenizer(line['summary'], max_length=self.summary_len, pad_to_max_length=True, truncation=True,
                                return_tensors='pt')
        source_ids = source.input_ids.squeeze()
        source_mask = source.attention_mask.squeeze()
        target_ids = target.input_ids.squeeze()
        target_mask = target.attention_mask.squeeze()
        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }

    def __iter__(self):
        return cycle(self.parse_file())


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    training_set = CustomDataset('../data_generator/train_section_slides_t5.json',
                                 tokenizer, 10, 5)
    train_params = {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 1
    }
    dataloader = DataLoader(training_set, **train_params)

    for batch in dataloader:
        print(batch)
