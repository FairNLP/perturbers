from perturbers.training.core import train_perturber
from perturbers.training.utils import TrainingConfig


def main():
    test_config = TrainingConfig(
        model_name="facebook/bart-large",
        train_steps=20000,  # TODO set to 10 epochs
        val_steps=1000,
        use_wandb=True,
        use_gpu=True,
        max_length=512,
        train_batch_size=16,
        test_batch_size=16,
        accumulate_grad_batches=4,
        es_patience=5,
        version="original-perturber",
        sep_token=", ",
        pert_sep_token=" <PERT_SEP> ",
    )

    train_perturber(test_config)


if __name__ == "__main__":
    main()
