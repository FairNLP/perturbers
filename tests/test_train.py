from perturbers.training.core import train_perturber
from perturbers.training.utils import TrainingConfig

test_config = TrainingConfig(
    model_name="hf-internal-testing/tiny-random-bart",
    debug=True,
    max_length=64,
)


def test_train_perturber():
    train_perturber(test_config)
