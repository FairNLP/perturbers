import random
from dataclasses import dataclass
from typing import Optional

from transformers import BartForConditionalGeneration, AutoTokenizer

from perturbers.data.panda_dict import get_panda_dict


@dataclass
class PerturberConfig:
    sep_token: str = '<SEP>'
    pert_sep_token: str = '<PERT_SEP>'
    max_length: int = 128


class Perturber:

    def __init__(self, model=None, config: Optional[PerturberConfig] = None):
        self.config = config if config is not None else PerturberConfig()

        if model is None:
            model_name = "facebook/perturber"
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            self.config.sep_token = ","
            self.config.pert_sep_token = "<PERT_SEP>"
        else:
            model_name = model.config.name_or_path
            self.model = model
        self.model.config.max_length = self.config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.tokenizer.add_tokens([self.config.pert_sep_token], special_tokens=True)

        self.panda_dict = get_panda_dict()
        self.input_template = PerturberTemplate(sep=self.config.sep_token, pert_sep=self.config.pert_sep_token,
                                                original=model_name == "facebook/perturber")

    def generate(self, input_txt: str, word: str = "", attribute: str = "", tokenizer_kwargs=None) -> str:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        input_txt = self.input_template(input_txt, word, attribute)
        output_tokens = self.model.generate(**self.tokenizer(input_txt, return_tensors='pt'), **tokenizer_kwargs)
        return self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            max_new_tokens=self.model.config.max_length
        )[0].lstrip()

    def __call__(self, input_txt, mode='word_list', tokenizer_kwargs=None, retry_unchanged=False):

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        if mode == 'highest_prob':
            raise NotImplementedError  # TODO
        elif mode == 'word_list':
            targets = [w for w in input_txt.split(" ") if w in self.panda_dict]
            perturbations = [(t, perturbed) for t in targets for perturbed in self.panda_dict[t]]
            random.shuffle(perturbations)
            for word, attribute in perturbations:
                generated_txt = self.generate(input_txt, word=word, attribute=attribute,
                                              tokenizer_kwargs=tokenizer_kwargs)
                if generated_txt != input_txt or not retry_unchanged:
                    return generated_txt
        else:
            raise NotImplementedError

        return input_txt


class PerturberTemplate:

    def __init__(self, sep: str = ",", pert_sep: str = "<PERT_SEP>", original: bool = False):
        self.sep = sep
        self.pert_sep = pert_sep if not original else f" {pert_sep}"

    def __call__(self, input_txt: str, word: str = "", attribute: str = "") -> str:
        return f"{word}{self.sep} {attribute}{self.pert_sep} {input_txt}"
