from datasets import concatenate_datasets, load_dataset

GENDER_ATTRIBUTES = {"man", "woman", "non-binary"}
RACE_ATTRIBUTES = {"black", "white", "hispanic", "native-american", "pacific-islander"}
AGE_ATTRIBUTES = {"child", "young", "middle-aged", "senior", "adult"}
ALL_ATTRIBUTES = GENDER_ATTRIBUTES | RACE_ATTRIBUTES | AGE_ATTRIBUTES


def get_panda_dict() -> dict[str, list[str]]:
    """
    Returns a dictionary of words and their associated attributes from the PANDA dataset. This can then serve as
    frequency-weighted word list for perturbation targets in open text.

    Returns:
        A dictionary of words and their associated attributes from the PANDA dataset.
    """
    dataset = load_dataset("facebook/panda")
    dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    perturbation_dict = {}
    for attribute, word in zip(dataset['target_attribute'], dataset['selected_word']):
        perturbation_dict[word] = perturbation_dict.get(word, []) + [attribute]
    sorted_dict = dict(reversed(sorted(perturbation_dict.items(), key=lambda item: len(item[1]))))
    return sorted_dict
