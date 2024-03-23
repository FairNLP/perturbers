from perturbers import Perturber
from perturbers.modeling.perturber import PerturberConfig


def test_word_list():
    model = Perturber("hf-internal-testing/tiny-random-bart")
    model(
        mode="word_list",
        input_txt="a",
        generate_kwargs={"max_new_tokens": 2}
    )


def test_classify():
    model = Perturber(
        model="hf-internal-testing/tiny-random-bart",
        config=PerturberConfig(conditional=False, max_length=8),
    )
    model(
        mode="classify",
        input_txt="a",
        generate_kwargs={"max_new_tokens": 2}
    )
