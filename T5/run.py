import argparse
import jsonlines
import logging
import nltk
import os
import pytorch_lightning as pl
import time
from pytorch_lightning.loggers import WandbLogger

from callbacks import LoggingCallback
from data_set import CustomDataset, StaticDataset
from fine_tune_t5_lightling import T5FineTuner

train = True

logger = logging.getLogger(__name__)
wandb_logger = WandbLogger(project='t5_slide_gen')
args_dict = dict(
    output_dir="",  # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_input_length=512,
    max_output_length=125,
    n_gpu=1,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    freeze_encoder=False,
    freeze_embeds=False,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=4,
    gradient_accumulation_steps=1,
    resume_from_checkpoint="t5_checkpoint/checkpoint-epoch=5-step=164581.ckpt",
    val_check_interval=0.03,
    early_stop_callback=False,
    opt_level='O1',
    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

if __name__ == '__main__':
    if train:
        args_dict.update(
            {'output_dir': 't5_checkpoint', 'num_train_epochs': 10, 'train_batch_size': 1, 'eval_batch_size': 1})
        args = argparse.Namespace(**args_dict)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(args.output_dir, prefix="checkpoint", monitor="val_loss",
                                                           mode="min", save_top_k=3)
        # If resuming from checkpoint, add an arg resume_from_checkpoint
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            # early_stop_callback=False,
            amp_level=args.opt_level,
            # resume_from_checkpoint=args.resume_from_checkpoint,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            val_check_interval=args.val_check_interval,
            # accelerator='ddp_spawn',
            # plugins='ddp_sharded',
            logger=wandb_logger,
            callbacks=[LoggingCallback(logger=logger)],
        )
        model = T5FineTuner(args)
        # import pickle
        # with open("tmp_file.pk", "wb") as f:
        #     pickle.dump(model, f)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

    if not train:
        test_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=1,
            resume_from_checkpoint=args.resume_from_checkpoint,
            logger=wandb_logger,
            callbacks=[LoggingCallback(logger=logger)],
        )
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

        model = T5FineTuner(args)
        trainer = pl.Trainer(**test_params)
        trainer.test(model, test_dataloaders=None, ckpt_path='best', verbose=True, datamodule=None)
