import random
from dataclasses import dataclass
from typing import Optional, Union

from transformers import BartForConditionalGeneration, AutoTokenizer, PreTrainedModel

from perturbers.data.panda_dict import get_panda_dict, ALL_ATTRIBUTES


@dataclass
class PerturberConfig:
    sep_token: str = '<SEP>'
    pert_sep_token: str = '<PERT_SEP>'
    max_length: int = 128


class Perturber:

    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, str]] = None,
            config: Optional[PerturberConfig] = None
    ) -> None:
        """
        Initializes the perturber with a model and configuration.

        Args:
            model: The model to be used for perturbation. This can either be an instance of a PreTrainedModel or a
                string representing the model name. If not provided, the original perturber model is used.

            config: An instance of the PerturberConfig class that contains configuration parameters for the
                perturber. If not provided, the default configuration is used.
        """

        self.config = config if config is not None else PerturberConfig()

        if isinstance(model, PreTrainedModel):
            model_name = model.config.name_or_path
            self.model = model
        elif isinstance(model, str):
            model_name = model
            self.model = BartForConditionalGeneration.from_pretrained(model)
        else:
            model_name = "facebook/perturber"
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            self.config.sep_token = ","
            self.config.pert_sep_token = "<PERT_SEP>"

        self.model.config.max_length = self.config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.tokenizer.add_tokens([self.config.pert_sep_token], special_tokens=True)

        self.panda_dict = get_panda_dict()
        self.input_template = PerturberTemplate(sep=self.config.sep_token, pert_sep=self.config.pert_sep_token,
                                                original=model_name == "facebook/perturber")

    def generate(self, input_txt: str, word: str = "", attribute: str = "", tokenizer_kwargs=None) -> str:
        """
        Generates a perturbed version of the input text.

        Args:
            input_txt: String to be perturbed

            word: The word of input_txt to be perturbed

            attribute: The attribute of the word to be perturbed

            tokenizer_kwargs: Additional keyword arguments to be passed to the tokenizer

        Returns:
            Perturbed version of the input text
        """

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if attribute and attribute not in ALL_ATTRIBUTES:
            raise ValueError(f"Attribute {attribute} not in {ALL_ATTRIBUTES}")
        input_txt = self.input_template(input_txt, word, attribute)
        output_tokens = self.model.generate(**self.tokenizer(input_txt, return_tensors='pt'), **tokenizer_kwargs)
        return self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            max_new_tokens=self.model.config.max_length
        )[0].lstrip()

    def __call__(self, input_txt, mode='word_list', tokenizer_kwargs=None, retry_unchanged=False
                 ) -> Union[str, NotImplementedError]:
        """
        Perturbs the input text using the specified mode and returns the perturbed text. No target word or attribute
        needs to be specified for this method.

        Args:
            input_txt: The input text to be perturbed

            mode: The mode to be used for perturbation. Currently, only 'word_list' is supported

            tokenizer_kwargs: Additional keyword arguments to be passed to the tokenizer

            retry_unchanged: If True, perturbation is retried with different target words/attributes until the output is
            different from the input

        Returns:
            Perturbed version of the input text
        """
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
    """
    A template for generating perturbed text from input text, word and attribute. A distinction is made between the
    template of the original paper and the template used to train models with this library, which use the <SEP> token
    instead of a comma. Words are prefixed with a space so that the target word and attribute have the same token as
    their occurrences in the input text.
    """

    def __init__(self, sep: str = ",", pert_sep: str = "<PERT_SEP>", original: bool = False) -> None:
        self.sep = sep
        self.pert_sep = pert_sep if not original else f" {pert_sep}"

    def __call__(self, input_txt: str, word: str = "", attribute: str = "") -> str:
        return f"{word}{self.sep} {attribute}{self.pert_sep} {input_txt}"
