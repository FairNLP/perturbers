import os
from typing import List

import lightning
import torch
from datasets import load_dataset
from lightning import Trainer, seed_everything
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity, BLEUScore
from transformers import AutoModel, BartForConditionalGeneration  # noqa 401
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from perturbers.modeling.perturber import PerturberTemplate
from perturbers.training.utils import TrainingConfig, get_diff_indices


class LightningWrapper(lightning.LightningModule):

    def __init__(self, c: TrainingConfig, tokenizer):
        super().__init__()
        # self.model = AutoModel.from_pretrained(c.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(c.model_name)
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer

        self._device = "cuda" if (c.use_gpu and torch.cuda.is_available()) else "cpu"

        self.learning_rate = c.learning_rate
        self.num_steps = c.train_steps
        self.train_batch_size = c.train_batch_size
        self.test_batch_size = c.test_batch_size
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self.train_metrics = self.get_metric_dict(c, "train")
        self.val_metrics = self.get_metric_dict(c, "val")
        self.test_metrics = self.get_metric_dict(c, "test")

    def get_metric_dict(self, c: TrainingConfig, split: str):
        metrics = {
            f'{split}_ppl': Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device),
            f'{split}_ppl_perturbed': Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device),
        }
        if split == "test":
            metrics[f'{split}_bleu4'] = BLEUScore(n_gram=4).to(self._device)
        return metrics

    def update_metrics(self, batch, outputs, metrics, generations=None):
        for metric_key, metric in metrics.items():
            if "bleu" in metric_key and generations is not None:
                value = metric(
                    preds=generations,
                    target=[[_] for _ in self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)],
                )
            elif "ppl" in metric_key:
                if "perturbed" in metric_key:
                    idx = batch["perturbed_idx"]
                    value = metric(preds=outputs[idx].unsqueeze(0), target=batch['labels'][idx].unsqueeze(0))
                else:
                    value = metric(preds=outputs, target=batch['labels'])
            else:
                raise NotImplementedError(f"Unsupported metric key: {metric_key}")
            self.log(metric_key, value=value, on_step=metric_key.startswith("train"), on_epoch=True, prog_bar=True,
                     batch_size=self.train_batch_size if metric_key.startswith("train") else self.test_batch_size)

    @staticmethod
    def clear_metrics(metrics):
        for metric_key, metric in metrics.items():
            metric.reset()

    def training_step(self, batch, batch_idx):
        outputs, loss = self.forward(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.train_batch_size)
        self.update_metrics(batch, outputs, self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self.forward(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.test_batch_size)
        self.update_metrics(batch, outputs, self.val_metrics)
        return loss

    def test_step(self, batch, batch_idx):
        outputs, loss = self.forward(batch)
        generations = self.generate(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=self.test_batch_size)
        self.update_metrics(batch, outputs, self.test_metrics, generations)
        return loss

    def on_train_epoch_end(self) -> None:
        self.clear_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self.clear_metrics(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self.clear_metrics(self.test_metrics)

    def forward(self, batch):
        outputs = self.model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]})
        return outputs.logits, outputs.loss

    def generate(self, batch):
        generations = self.model.generate(
            **{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]},
            max_length=batch['input_ids'].shape[-1],
        )
        return self.tokenizer.batch_decode(generations, skip_special_tokens=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[p for p in self.model.parameters()],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.num_steps,
            num_warmup_steps=int(0.1 * self.num_steps)
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def get_collate_fn(c: TrainingConfig, tokenizer, tokenizer_kwargs):
    input_template = PerturberTemplate(sep=c.sep_token, pert_sep=c.pert_sep_token,
                                       original=c.model_name == "facebook/perturber")

    def collate_fn(batch: List):
        original, perturbed = [], []
        perturbed_x, perturbed_y = [], []
        for i, item in enumerate(batch):
            perturbed.append(item['perturbed'])
            original.append(input_template(item["original"], item["selected_word"], item["target_attribute"]))
            idx = item["perturbed_idx"]
            perturbed_x += [i] * len(idx)
            perturbed_y += idx

        original = tokenizer(original, return_tensors='pt', **tokenizer_kwargs)
        perturbed = tokenizer(perturbed, return_tensors='pt', **tokenizer_kwargs)

        return {
            "input_ids": original["input_ids"],
            "attention_mask": original["attention_mask"],
            "labels": perturbed["input_ids"],
            "perturbed_idx": (perturbed_x, perturbed_y),
        }

    return collate_fn


def get_loggers(c: TrainingConfig):
    loggers = [CSVLogger(save_dir=c.save_path, name=c.version)]
    if c.use_wandb:
        loggers.append(WandbLogger(name=c.version, save_dir=c.save_path, version=c.version, project="perturbers"))
    return loggers


def get_callbacks(c: TrainingConfig):
    return [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=c.es_delta,
            patience=c.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_on_train_epoch_end=False,
            dirpath=os.path.join(c.save_path, c.version),
            every_n_epochs=1
        ),
    ]


def add_indices(sample, tokenizer, tokenizer_kwargs):
    sample["perturbed_idx"] = get_diff_indices(
        tokenizer(sample['original'], **tokenizer_kwargs).data['input_ids'],
        tokenizer(sample['perturbed'], **tokenizer_kwargs).data['input_ids']
    )
    return sample


def train_perturber(c: TrainingConfig):
    seed_everything(c.seed, workers=True)

    if c.debug:
        c.train_steps = 10
        c.val_steps = 5
        c.accumulate_grad_batches = 1

    tokenizer = AutoTokenizer.from_pretrained(c.model_name, add_prefix_space=True)
    tokenizer.add_tokens([c.sep_token, c.pert_sep_token], special_tokens=True)
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": c.max_length}
    model = LightningWrapper(c, tokenizer)
    dataset = load_dataset(c.dataset_name)

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if c.debug:
        train_ds = train_ds.select(range(128))
        val_ds = val_ds.select(range(128))

    train_ds = train_ds.map(lambda x: add_indices(x, tokenizer, tokenizer_kwargs), num_proc=max(c.num_workers, 1))
    val_ds = val_ds.map(lambda x: add_indices(x, tokenizer, tokenizer_kwargs), num_proc=max(c.num_workers, 1))

    collate_fn = get_collate_fn(c, tokenizer, tokenizer_kwargs)

    dl_kwargs = {"collate_fn": collate_fn, "num_workers": c.num_workers}
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=c.train_batch_size, **dl_kwargs)
    val_dataloader = DataLoader(val_ds, shuffle=False, batch_size=c.test_batch_size, **dl_kwargs)

    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and c.use_gpu) else "cpu",
        enable_checkpointing=True,
        max_steps=c.train_steps,
        val_check_interval=c.val_steps,
        callbacks=get_callbacks(c),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if c.fp16 else 32,
        logger=get_loggers(c),
        check_val_every_n_epoch=None,
        gradient_clip_val=c.gradient_clipping_value,
        accumulate_grad_batches=c.accumulate_grad_batches,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(
        dataloaders=val_dataloader,
        ckpt_path='best',
    )

    if c.output_path:
        model.model.save_pretrained(c.output_path)
        tokenizer.save_pretrained(c.output_path)

    if c.push_to_hub:
        model.model.push_to_hub(c.hub_repo_id)
        tokenizer.push_to_hub(c.hub_repo_id)

    return model.model
