import json
import os
import torch
from transformers import (
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.utils import logging
from typing import Dict, Iterator

logger = logging.get_logger(__name__)


class Seq2SeqTrainerAdvance(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (:obj:`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{v >> 20}MB"
            elif k == "total_flos":
                metrics_copy[k] = f"{int(v) >> 30}GF"
            elif type(metrics_copy[k]) == float:
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def log_metrics(self, split, metrics):
        """
        Log metrics in a specially formatted way

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predictmetrics: metrics dict
        """
        if not self.is_world_process_zero():
            return

        logger.info(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. ``train_results.json``.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (:obj:`str`):
                Mode/split name: one of ``train``, ``eval``, ``test``, ``all``
            metrics (:obj:`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Creates combined metrics by updating ``all_results.json`` with metrics of this call
        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

    def save_state(self):
        """
        Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

        Under distributed environment this is done only for a process with rank 0.
        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, "trainer_state.json")
        self.state.save_to_json(path)
