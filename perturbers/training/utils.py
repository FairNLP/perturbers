from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class TrainingConfig:
    model_name: str = "facebook/bart-large"
    dataset_name: str = "facebook/panda"
    train_batch_size: int = 64
    test_batch_size: int = 64
    max_length: int = 512
    learning_rate: float = 1e-5
    tokenizer: Optional[str] = None
    save_path: str = "output"
    es_patience: int = 10
    es_delta: float = 0.0
    use_gpu: bool = True
    use_wandb: bool = False
    version: str = field(default_factory=lambda: datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    train_steps: int = 10
    val_steps: int = 5
    fp16: bool = False
    seed: int = 42
    sep_token: str = '<SEP>'
    pert_sep_token: str = '<PERT_SEP>'
    debug: bool = False
    gradient_clipping_value: float = 0.1
    accumulate_grad_batches: int = 0
    output_path: str = ""
    push_to_hub: bool = False
    hub_repo_id: str = ""
    num_workers: int = 0


def get_diff_indices(tokens1, tokens2):
    matcher = SequenceMatcher(None, tokens1, tokens2)
    diff_indices = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # For each block that is not equal, add the indices from tokens2
            diff_indices.extend(range(j1, j2))

    return diff_indices
