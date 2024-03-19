from perturbers import Perturber

UNPERTURBED = "Jack was passionate about rock climbing and his love for the sport was infectious to all men around him."
PERTURBED_SMALL = "Mary was passionate about rock climbing and her love for the sport was infectious to all men around her."
PERTURBED_BASE = "Jane was passionate about rock climbing and her love for the sport was infectious to all men around her."


def test_small_perturber_model():
    model = Perturber("fairnlp/unconditional-perturber-small")

    perturbed, probability = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_SMALL


def test_base_perturber_model():
    model = Perturber("fairnlp/unconditional-perturber-base")

    perturbed, probability = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_BASE
