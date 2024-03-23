from perturbers import Perturber
from perturbers.data.panda_dict import GENDER_ATTRIBUTES, ALL_ATTRIBUTES, attribute_to_token

UNPERTURBED = "Jack was passionate about rock climbing and his love for the sport was infectious to all men around him."
PERTURBED_SMALL = "Mary was passionate about rock climbing and her love for the sport was infectious to all men around her."
PERTURBED_BASE = "Jacqueline was passionate about rock climbing and her love for the sport was infectious to all men around her."


def test_small_perturber_model():
    model = Perturber("fairnlp/unconditional-perturber-small")

    perturbed, probability = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_SMALL


def test_base_perturber_model():
    model = Perturber("fairnlp/unconditional-perturber-base")

    perturbed, probability = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_BASE


def test_get_attribute_probabilities():
    model = Perturber("fairnlp/unconditional-perturber-small")
    gender_token_map = {a: attribute_to_token(a) for a in GENDER_ATTRIBUTES}
    male_probabilities = model.get_attribute_probabilities("He", attributes=gender_token_map)
    female_probabilities = model.get_attribute_probabilities("She", attributes=gender_token_map)
    assert male_probabilities["woman"] > female_probabilities["woman"]
    assert female_probabilities["man"] > male_probabilities["man"]
    assert len(male_probabilities) == len(gender_token_map)

    all_probabilities = model.get_attribute_probabilities("")
    assert sum(list(all_probabilities.values())) - 1 < 1e-6
    assert len(all_probabilities) == len(ALL_ATTRIBUTES)
