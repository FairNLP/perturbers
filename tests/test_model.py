from perturbers import Perturber

UNPERTURBED = "Jack was passionate about rock climbing and his love for the sport was infectious to all men around him."
PERTURBED_ORIGINAL = "Jane was passionate about rock climbing and her love for the sport was infectious to all men around her."
PERTURBED_SMALL = "Jack was passionate about rock climbing and her love for the sport was infectious to all men around her."
PERTURBED_BASE = "Jacqueline was passionate about rock climbing and her love for the sport was infectious to all men around her."


def test_perturber_model():
    model = Perturber()

    perturbed = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_ORIGINAL


def test_small_perturber_model():
    model = Perturber("fairnlp/perturber-small")

    perturbed = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_SMALL


def test_base_perturber_model():
    model = Perturber("fairnlp/perturber-base")

    perturbed = model.generate(UNPERTURBED, "Jack", "woman")
    assert perturbed == PERTURBED_BASE
