from typing import List

from perturbers.data.panda_dict import attribute_to_token
from perturbers.modeling.perturber import PerturberTemplate
from perturbers.training.core import preprocess_inputs, get_tokenizer
from perturbers.training.utils import TrainingConfig

# mock data
sample = {
    "original": "Perturbers are cool!",
    "perturbed": "Perturbers are great!",
    "selected_word": "cool",
    "target_attribute": "non-binary",
}
perturbed_span = " great"
selected_word_span = " cool"
target_attribute_span = attribute_to_token("non-binary")


def get_preprocessed_sample(c: TrainingConfig) -> dict:
    tokenizer, tokenizer_kwargs = get_tokenizer(c)
    input_template = PerturberTemplate(sep=c.sep_token, pert_sep=c.pert_sep_token,
                                       original=c.model_name == "facebook/perturber", conditional=c.conditional)
    preprocessed = preprocess_inputs(
        sample=sample,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        c=c,
        input_template=input_template,
        tokenize=False,
    )
    return preprocessed


def get_span_at_idx(sequence: str, token_idx: List[int], c: TrainingConfig) -> str:
    tokenizer, tokenizer_kwargs = get_tokenizer(c)
    tokens = tokenizer(sequence, **tokenizer_kwargs)
    return tokenizer.decode([tokens['input_ids'][i] for i in token_idx])


def test_prepare_inputs_conditional():
    c = TrainingConfig(
        model_name="hf-internal-testing/tiny-random-bart",
        debug=True,
        max_length=64,
        conditional=True,
    )
    preprocessed = get_preprocessed_sample(c)
    span = get_span_at_idx(
        sequence=preprocessed["perturbed"],
        token_idx=preprocessed["perturbed_idx"],
        c=c,
    )
    assert span == perturbed_span


def test_prepare_inputs_unconditional():
    c = TrainingConfig(
        model_name="hf-internal-testing/tiny-random-bart",
        debug=True,
        max_length=64,
        conditional=False,
    )
    preprocessed = get_preprocessed_sample(c)
    assert get_span_at_idx(
        sequence=preprocessed["perturbed"],
        token_idx=preprocessed["perturbed_idx"],
        c=c,
    ) == perturbed_span

    assert get_span_at_idx(
        sequence=preprocessed["perturbed"],
        token_idx=preprocessed["word_idx"],
        c=c,
    ) == selected_word_span

    assert get_span_at_idx(
        sequence=preprocessed["perturbed"],
        token_idx=preprocessed["attribute_idx"],
        c=c,
    ) == target_attribute_span
