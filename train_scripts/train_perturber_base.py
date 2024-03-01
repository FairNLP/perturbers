from datetime import datetime

from perturbers.training.core import train_perturber
from perturbers.training.utils import TrainingConfig


def main():
    test_config = TrainingConfig(
        model_name="facebook/bart-base",
        train_steps=20000,
        val_steps=1000,
        use_wandb=True,
        use_gpu=True,
        max_length=512,
        train_batch_size=16,
        test_batch_size=16,
        accumulate_grad_batches=4,
        es_patience=5,
        version=f"perturber-base-{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        learning_rate=1e-5,
        push_to_hub=True,
        hub_repo_id="fairnlp/perturber-base",
        num_workers=7,
    )

    train_perturber(test_config)


if __name__ == "__main__":
    main()
