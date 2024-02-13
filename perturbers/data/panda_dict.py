from datasets import concatenate_datasets, load_dataset


def get_panda_dict():
    dataset = load_dataset("facebook/panda")
    dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    perturbation_dict = {}
    for attribute, word in zip(dataset['target_attribute'], dataset['selected_word']):
        perturbation_dict[word] = perturbation_dict.get(word, []) + [attribute]
    sorted_dict = dict(reversed(sorted(perturbation_dict.items(), key=lambda item: len(item[1]))))
    return sorted_dict
