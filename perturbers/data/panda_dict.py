from typing import Optional, List

from datasets import concatenate_datasets, load_dataset

GENDER_ATTRIBUTES = {"man", "woman", "non-binary"}
RACE_ATTRIBUTES = {"black", "white", "asian", "hispanic", "native-american", "pacific-islander"}
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


def get_attribute_tokens(attribute_set: Optional[set[str]] = None) -> List[str]:
    """
    Creates specific attribute tokens

    Args:
        attribute_set: Set of attributes to be used for token generation. If None, all attributes are used.

    Returns:
        A list of attribute tokens.
    """
    if attribute_set is None:
        attribute_set = ALL_ATTRIBUTES

    return [attribute_to_token(attr) for attr in attribute_set]


def attribute_to_token(attribute: str) -> str:
    """
    Converts an attribute to a token

    Args:
        attribute: The attribute to be converted to a token

    Returns:
        The token corresponding to the attribute
    """
    return f"<{attribute.upper().replace('-', '_').replace(' ', '_')}>"
