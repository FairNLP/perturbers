from perturbers import Perturber


def test_word_list():
    model = Perturber("hf-internal-testing/tiny-random-bart")
    model(
        mode="word_list",
        input_txt="a",
        generate_kwargs={"max_new_tokens": 2}
    )


def test_highest_prob():
    model = Perturber("hf-internal-testing/tiny-random-bart")
    model(
        mode="classify",
        input_txt="a",
        generate_kwargs={"max_new_tokens": 2}
    )
