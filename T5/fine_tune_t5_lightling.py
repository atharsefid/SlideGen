import datasets
import numpy as np
import pytorch_lightning as pl
import time
import torch
from nlp import load_metric
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from typing import Optional, Callable

from data_set import CustomDataset


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        # if 't5-small' == hparams.model_name_or_path:
        #     self.device_map = {0: [0, 1],
        #                        1: [2, 3],
        #                        2: [4],
        #                        3: [5]}
        # elif hparams.model_name_or_path == 't5-base':
        #     self.device_map = {0: [0, 1, 2],
        #                        1: [3, 4, 5],
        #                        2: [6, 7, 8],
        #                        3: [9, 10, 11]}
        # self.model.parallelize(self.device_map)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        # self.rouge_metric = load_metric('rouge')
        self.preds = []
        self.targets = []
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            self.freeze_params(self.model.get_encoder())

    @staticmethod
    def freeze_params(model):
        for par in model.parameters():
            par.requires_grad = False

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            self.freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                self.freeze_params(d.embed_positions)
                self.freeze_params(d.embed_tokens)
        except AttributeError:
            self.freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                self.freeze_params(d.embed_tokens)

    @staticmethod
    def list_map(f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    @staticmethod
    def parse_score(result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        # lm_labels = batch["target_ids"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            labels=batch["target_ids"]
        )
        return outputs[0]

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.list_map(str.strip, gen_text)

    def _generative_val_step(self, batch):

        t0 = time.time()
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=250,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])

        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]

        loss = self._step(batch)
        base_metrics = {'val_loss': loss}
        #         rouge: Dict = self.calc_generative_metrics(preds, target)
        summary_len = np.mean(self.list_map(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summary_len, preds=preds, target=target)
        # self.rouge_metric.add_batch(preds, target)
        self.preds.append(' '.join(preds))
        self.targets.append(' '.join(target))

        return base_metrics

    def _generative_test_step(self, batch):
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=250,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        self.preds.append(' '.join(preds))
        self.targets.append(' '.join(target))
        gen_time = (time.time() - t0) / batch["source_ids"].shape[0]
        loss = self._step(batch)
        base_metrics = {'test_loss': loss}
        summary_len = np.mean(self.list_map(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summary_len, preds=preds, target=target)
        self.output_file.write(' '.join(preds) + '\n')
        return base_metrics

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        metrics = self._generative_val_step(batch)
        print('~~~~ val loss:', metrics['val_loss'])
        self.log('val_loss', metrics['val_loss'])
        return metrics
        # return { "log": {"val_loss":metrics['val_loss']}}

    def validation_epoch_end(self, validation_step_outputs):
        print('***** val epoch end:', validation_step_outputs)
        avg_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        rouge = datasets.load_metric('rouge')
        rouge_dict = rouge.compute(predictions=self.preds, references=self.targets)
        tensorboard_logs.update(rouge1=rouge_dict['rouge1'].mid.recall, rougeL=rouge_dict['rougeL'].mid.recall)
        # Clear out the lists for next epoch
        self.preds = []
        self.targets = []

        return {"avg_val_loss": avg_loss,
                "rouge1": rouge_dict['rouge1'],
                "rougeL": rouge_dict['rougeL'],
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        metrics = self._generative_test_step(batch)
        # self.log('test_loss', metrics['test_loss'])
        return metrics

    def test_epoch_end(self, test_step_outputs):
        with open('test_generated_text.txt', 'w')  as outfile:
            for pred in self.preds:
                outfile.write(pred + '\n')

        avg_loss = torch.stack([x["test_loss"] for x in test_step_outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss}
        rouge = datasets.load_metric('rouge')
        rouge_dict = rouge.compute(predictions=self.preds, references=self.targets)
        tensorboard_logs.update(rouge1=rouge_dict['rouge1'].mid.recall, rougeL=rouge_dict['rougeL'].mid.recall)
        # Clear out the lists for next epoch
        self.preds = []
        self.targets = []

        return {"avg_test_loss": avg_loss,
                "rouge1": rouge_dict['rouge1'],
                "rougeL": rouge_dict['rougeL'],
                "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int,
                       optimizer_closure: Optional[Callable] = None, on_tpu: bool = False,
                       using_native_amp: bool = False, using_lbfgs: bool = False, ):
        gradient_accumulation_steps = 1
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            self.lr_scheduler.step()
            optimizer.step(closure=optimizer_closure)

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):

        # training_set = CustomDataset('../data_generator/train_section_slides_t5.json',
        training_set = CustomDataset('../data_generator/train_section_slides_t5.json',
                                     self.tokenizer,
                                     self.hparams.max_input_length,
                                     self.hparams.max_output_length)
        train_params = {
            'batch_size': self.hparams.train_batch_size,
            'shuffle': False,
            'num_workers': 1
        }
        dataloader = DataLoader(training_set, **train_params)

        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        # val_section_slides_t5
        validating_set = CustomDataset('../data_generator/val_section_slides_t5.json',
                                       self.tokenizer,
                                       self.hparams.max_input_length,
                                       self.hparams.max_output_length)
        val_params = {
            'batch_size': self.hparams.eval_batch_size,
            'shuffle': False,
            'num_workers': 1
        }
        return DataLoader(validating_set, **val_params)

    def test_dataloader(self):
        # test_section_slides_t5
        testing_set = CustomDataset('../data_generator/test_section_slides_t5.json',
                                    self.tokenizer,
                                    self.hparams.max_input_length,
                                    self.hparams.max_output_length)
        test_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1
        }
        return DataLoader(testing_set, **test_params)
