import os
from typing import List, Tuple, Optional

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
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from perturbers.data.panda_dict import get_attribute_tokens
from perturbers.modeling.perturber import PerturberTemplate
from perturbers.training.utils import TrainingConfig, get_diff_indices


class LightningWrapper(lightning.LightningModule):
    """
    Wrapper class for the perturber model to be used with PyTorch Lightning.
    """

    def __init__(self, c: TrainingConfig, tokenizer: PreTrainedTokenizerBase) -> None:
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

    def get_metric_dict(self, c: TrainingConfig, split: str) -> dict[str, torch.nn.Module]:
        metrics = {
            f'{split}_ppl': Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device),
            f'{split}_ppl_perturbed': Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device),
        }
        if not c.conditional:
            metrics[f'{split}_ppl_word'] = Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device)
            metrics[f'{split}_ppl_attribute'] = Perplexity(ignore_index=self.tokenizer.pad_token_id).to(self._device)
        if split == "test":
            metrics[f'{split}_bleu4'] = BLEUScore(n_gram=4).to(self._device)
        return metrics

    def update_metrics(
            self,
            batch: dict,
            outputs: dict,
            metrics: dict[str, torch.nn.Module],
            generations: Optional[List[str]] = None
    ) -> None:
        for metric_key, metric in metrics.items():
            if "bleu" in metric_key and generations is not None:
                value = metric(
                    preds=generations,
                    target=[[_] for _ in self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)],
                )
            elif "ppl" in metric_key:
                idx = None
                if metric_key.endswith("perturbed"):
                    idx = batch["perturbed_idx"]
                elif metric_key.endswith("word"):
                    idx = batch["word_idx"]
                elif metric_key.endswith("attribute"):
                    idx = batch["attribute_idx"]
                if idx is not None:
                    value = metric(preds=outputs[idx].unsqueeze(0), target=batch['labels'][idx].unsqueeze(0))
                else:
                    value = metric(preds=outputs, target=batch['labels'])
            else:
                raise NotImplementedError(f"Unsupported metric key: {metric_key}")
            self.log(metric_key, value=value, on_step=metric_key.startswith("train"), on_epoch=True, prog_bar=True,
                     batch_size=self.train_batch_size if metric_key.startswith("train") else self.test_batch_size)

    @staticmethod
    def clear_metrics(metrics: dict[str, torch.nn.Module]):
        for metric_key, metric in metrics.items():
            metric.reset()

    def training_step(self, batch: dict):
        outputs, loss = self.forward(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=self.train_batch_size)
        self.update_metrics(batch, outputs, self.train_metrics)
        return loss

    def validation_step(self, batch: dict):
        outputs, loss = self.forward(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=self.test_batch_size)
        self.update_metrics(batch, outputs, self.val_metrics)
        return loss

    def test_step(self, batch: dict):
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

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]})
        return outputs.logits, outputs.loss

    def generate(self, batch: dict) -> List[str]:
        generations = self.model.generate(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        return self.tokenizer.batch_decode(generations, skip_special_tokens=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
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


def get_collate_fn(c: TrainingConfig, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict):
    """
    Get the collate function for the dataloaders.

    Args:
        c: The training configuration object

        tokenizer: The tokenizer to be used for tokenizing the input

        tokenizer_kwargs: Additional keyword arguments to be passed to the tokenizer

    Returns:
        The collate function for the dataloaders
    """

    def collate_fn(batch: List) -> dict:
        original, perturbed = [], []
        perturbed_x, perturbed_y = [], []
        attribute_x, attribute_y = [], []
        word_x, word_y = [], []
        for i, item in enumerate(batch):
            original.append(item["original"])
            perturbed.append(item["perturbed"])

            perturbed_idx = item["perturbed_idx"]
            perturbed_x += [i] * len(perturbed_idx)
            perturbed_y += perturbed_idx

            if not c.conditional:
                word_idx = item["word_idx"]
                word_x += [i] * len(word_idx)
                word_y += word_idx

                attribute_idx = item["attribute_idx"]
                attribute_x += [i] * len(attribute_idx)
                attribute_y += attribute_idx

        original = tokenizer(original, return_tensors='pt', **tokenizer_kwargs)
        perturbed = tokenizer(perturbed, return_tensors='pt', **tokenizer_kwargs)

        return_dict = {
            "input_ids": original["input_ids"],
            "attention_mask": original["attention_mask"],
            "labels": perturbed["input_ids"],
            "perturbed_idx": (perturbed_x, perturbed_y),
        }

        if not c.conditional:
            return_dict["word_idx"] = (word_x, word_y)
            return_dict["attribute_idx"] = (attribute_x, attribute_y)

        return return_dict

    return collate_fn


def get_loggers(c: TrainingConfig):
    loggers = [CSVLogger(save_dir=c.save_path, name=c.version)]
    if c.use_wandb:
        loggers.append(WandbLogger(name=c.version, save_dir=c.save_path, version=c.version, project="perturbers"))
    return loggers


def get_callbacks(c: TrainingConfig) -> List[pl.callbacks.Callback]:
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


def preprocess_inputs(sample: dict, tokenizer: PreTrainedTokenizerBase, tokenizer_kwargs: dict, c: TrainingConfig,
                      input_template: PerturberTemplate) -> dict:
    """
    Add the indices of the tokens that are different between the original and perturbed text to the sample dictionary.
    Function signature is intended to be used with the `map` method of the Hugging Face datasets library.
    """

    idx = get_diff_indices(
        tokenizer(sample['original'], **tokenizer_kwargs).data['input_ids'],
        tokenizer(sample['perturbed'], **tokenizer_kwargs).data['input_ids'],
    )

    if c.conditional:
        sample["perturbed_idx"] = idx
        sample['original'] = input_template(sample["original"], sample["selected_word"], sample["target_attribute"])
    else:

        # Account for prefix in perturbed indices
        sentence_prefix = input_template.get_sentence_prefix(sample["selected_word"], sample["target_attribute"])
        sentence_offset = len(tokenizer.tokenize(sentence_prefix))
        sample["perturbed_idx"] = [i + sentence_offset for i in idx]
        sample["perturbed"] = input_template(sample["perturbed"], sample["selected_word"], sample["target_attribute"])

        # Add indices for word and attribute
        n_cls_tokens = len(tokenizer.tokenize(tokenizer.bos_token))
        word_prefix = input_template.get_word_prefix(sample["selected_word"], sample["target_attribute"])
        word_offset = len(tokenizer.tokenize(word_prefix)) + n_cls_tokens
        sample["word_idx"] = [i + word_offset for i in range(len(tokenizer.tokenize(sample["selected_word"])))]

        attribute_prefix = input_template.get_attribute_prefix(sample["selected_word"], sample["target_attribute"])
        attribute_offset = len(tokenizer.tokenize(attribute_prefix)) + n_cls_tokens
        sample["attribute_idx"] = [attribute_offset]  # Only one attribute token

        for idx_key in ["perturbed_idx", "word_idx", "attribute_idx"]:
            sample[idx_key] = [i for i in sample[idx_key] if i < c.max_length]

    return sample


def train_perturber(c: TrainingConfig) -> PreTrainedModel:
    """
    Train a perturber model using the specified configuration.
    """
    seed_everything(c.seed, workers=True)

    if c.debug:
        c.train_steps = 10
        c.val_steps = 5
        c.accumulate_grad_batches = 1
        c.num_workers = 0

    tokenizer, tokenizer_kwargs = get_tokenizer(c)
    model = LightningWrapper(c, tokenizer)
    dataset = load_dataset(c.dataset_name)

    input_template = PerturberTemplate(sep=c.sep_token, pert_sep=c.pert_sep_token,
                                       original=c.model_name == "facebook/perturber", conditional=c.conditional)

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if c.debug:
        train_ds = train_ds.select(range(128))
        val_ds = val_ds.select(range(128))

    map_fn = lambda x: preprocess_inputs(x, tokenizer, tokenizer_kwargs, c, input_template)
    train_ds = train_ds.map(map_fn, num_proc=max(c.num_workers, 1))
    val_ds = val_ds.map(map_fn, num_proc=max(c.num_workers, 1))

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


def get_tokenizer(c):
    tokenizer = AutoTokenizer.from_pretrained(c.model_name, add_prefix_space=True)
    new_tokens = [c.sep_token, c.pert_sep_token]
    if not c.conditional:
        new_tokens += get_attribute_tokens()
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": c.max_length}
    return tokenizer, tokenizer_kwargs
