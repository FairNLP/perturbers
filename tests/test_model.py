from perturbers import Perturber


def test_perturber_model():
    model = Perturber()

    unperturbed = "Jack was passionate about rock climbing and his love for the sport was infectious to all men around him."
    perturbed = model.generate(unperturbed, "jack", "female")

    assert perturbed == "Jackie was passionate about rock climbing and her love for the sport was infectious to all men around her."

    assert model(unperturbed)
