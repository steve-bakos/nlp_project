import logging
from typing import Dict, List, Optional
from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
from transformers.file_utils import is_torch_tpu_available
import math

from datasets import Dataset
import time

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class TrainerWithSeveralEvalDatasets(Trainer):
    def __init__(
        self,
        *args,
        eval_datasets: Optional[List[Dataset]] = None,
        eval_prefixes: Optional[List[str]] = None,
        eval_aggregation=None,
        **kwargs,
    ):
        if eval_datasets is None:
            raise Exception(
                f"Not filling `eval_datasets` argument of TrainerWithSeveralEvalDatasets makes no sense, please use Trainer instead"
            )
        if kwargs.get("eval_dataset") is not None:
            logging.warning(
                "eval_dataset (without s) was given to TrainerWithSeveralEvalDatasets, it will be ignored (in favor of eval_datasets)"
            )
            del kwargs["eval_dataset"]
        if eval_prefixes is None:
            eval_prefixes = list(map(str, range(len(eval_datasets))))
        if len(eval_datasets) != len(eval_prefixes):
            raise Exception(
                f"TrainerWithSeveralEvalDatasets.__init__: Length of `eval_datasets` should match length of `eval_prefixes` if provided. got {len(eval_datasets)} and {len(eval_prefixes)}"
            )

        super().__init__(*args, **kwargs)

        self.eval_datasets = eval_datasets
        self.eval_prefixes = eval_prefixes
        self.eval_aggregation = eval_aggregation

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is not None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        self._memory_tracker.start()
        start_time = time.time()
        metrics = {}
        num_samples = 0
        for prefix, dataset in zip(self.eval_prefixes, self.eval_datasets):
            eval_dataloader = self.get_eval_dataloader(dataset)

            eval_loop = (
                self.prediction_loop
                if self.args.use_legacy_prediction_loop
                else self.evaluation_loop
            )
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix + f"_{prefix}",
            )
            metrics.update(output.metrics)
            num_samples += output.num_samples

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        if self.eval_aggregation is not None:
            metrics.update(self.eval_aggregation(metrics, metric_key_prefix))

        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
