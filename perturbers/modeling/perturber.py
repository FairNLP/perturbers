import logging
import random
from dataclasses import dataclass
from typing import Optional, Union, Literal, Tuple, List

import numpy as np
from transformers import BartForConditionalGeneration, AutoTokenizer, PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig

from perturbers.data.panda_dict import get_panda_dict, attribute_to_token, ALL_ATTRIBUTES


@dataclass
class PerturberConfig:
    sep_token: str = '<SEP>'
    pert_sep_token: str = '<PERT_SEP>'
    max_length: int = 128
    conditional: bool = True


@dataclass
class UnconditionalPerturberConfig(PerturberConfig):
    conditional: bool = False


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
        if config is None and "unconditional" in model_name:
            logging.info("Inferring unconditional perturber from model name")
            self.config.conditional = False

        self.attribute_to_token = {a: attribute_to_token(a) for a in ALL_ATTRIBUTES}
        self.token_to_attribute = {t: a for a, t in self.attribute_to_token.items()}
        self.attribute_tokens = {attribute_to_token(a) for a in ALL_ATTRIBUTES}

        self.model.config.max_length = self.config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.tokenizer.add_tokens([self.config.pert_sep_token], special_tokens=True)
        if self.model.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            logging.warning("Number of tokens in tokenizer does not match number of tokens in model. The model "
                            "embeddings will be resized by adding random weights.")
            self.model.model.resize_token_embeddings(len(self.tokenizer))

        self.panda_dict = get_panda_dict()
        self.input_template = PerturberTemplate(sep=self.config.sep_token, pert_sep=self.config.pert_sep_token,
                                                original=model_name == "facebook/perturber",
                                                conditional=self.config.conditional)

    def get_attribute_probabilities(self, input_txt: str):
        if self.config.conditional:
            raise RuntimeError("Attribute classification is not possible for conditional perturber models")
        # TODO unconditional perturber methods for classifying the attribute and
        attribute_tokens = []

    def generate_conditions(self, input_txt: str, n_permutations: int, tokenizer_kwargs: dict) -> List[tuple[str, str]]:
        if self.config.conditional:
            raise RuntimeError("Attribute classification is not possible for conditional perturber models")
        gen_config = GenerationConfig.from_model_config(self.model.config)
        gen_config.update(eos_token_id=self.tokenizer.vocab[self.config.pert_sep_token])
        generation = self.model.generate(
            **self.tokenizer(input_txt, return_tensors='pt', **tokenizer_kwargs),
            generation_config=gen_config,
            max_new_tokens=self.model.config.max_length,
            num_return_sequences=n_permutations,
            num_beams=n_permutations,
        )
        attribute_tokens = generation[:, 2]
        target_tokens = generation[:, 3:]

        # Hack to prevent double brackets from InputTemplate
        attributes = [self.token_to_attribute.get(a) for a in self.tokenizer.batch_decode(attribute_tokens,
                                                                                          max_new_tokens=self.model.config.max_length)]
        target_words = [w.lstrip() for w in self.tokenizer.batch_decode(target_tokens, skip_special_tokens=True,
                                                                        max_new_tokens=self.model.config.max_length)]
        # Filter attribute hallucinations (unlikely)
        return [(w, a) for idx, (w, a) in enumerate(zip(target_words, attributes)) if a is not None]

    def generate(self, input_txt: str, word: str = "", attribute: str = "", tokenizer_kwargs: Optional[dict] = None,
                 generate_kwargs: Optional[dict] = None) -> Tuple[str, float]:
        """
        Generates a perturbed version of the input text.

        Args:
            input_txt: String to be perturbed

            word: The word of input_txt to be perturbed

            attribute: The attribute of the word to be perturbed

            tokenizer_kwargs: Additional keyword arguments to be passed to the tokenizer

            generate_kwargs: Additional keyword arguments to be passed to the generate method

        Returns:
            Perturbed version of the input text along with the average token probability
        """

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if generate_kwargs is None:
            generate_kwargs = {}
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True

        # Validate the attribute -- generated attribute is validated after generation
        if self.config.conditional and attribute and attribute not in ALL_ATTRIBUTES:
            raise ValueError(f"Attribute {attribute} not in {ALL_ATTRIBUTES}")

        if self.config.conditional:
            input_txt = self.input_template(input_txt, word, attribute)
            tokens = self.tokenizer(input_txt, return_tensors='pt', **tokenizer_kwargs)
            outputs = self.model.generate(**tokens, **generate_kwargs)
        else:
            prefix = self.tokenizer.bos_token + self.input_template.get_sentence_prefix(word, attribute)
            encoder_tokens = self.tokenizer(input_txt, return_tensors='pt', **tokenizer_kwargs)
            decoder_tokens = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False, **tokenizer_kwargs)
            outputs = self.model.generate(
                input_ids=encoder_tokens.data["input_ids"],
                attention_mask=encoder_tokens.data["attention_mask"],
                decoder_input_ids=decoder_tokens.data["input_ids"],
                decoder_attention_mask=decoder_tokens.data["attention_mask"],
                **generate_kwargs
            )
        output_string = self._decode_generation(outputs)
        probabilities = self.model.compute_transition_scores(outputs.sequences, outputs.scores).exp()
        return output_string, float(probabilities.mean())

    def _decode_generation(self, outputs):
        if self.config.conditional:
            decode_tokens = outputs.sequences
        else:
            output_string = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=False, max_new_tokens=self.model.config.max_length
            )
            output_string = self.config.pert_sep_token.join(output_string.split(self.config.pert_sep_token)[1:])
            decode_tokens = self.tokenizer(output_string, return_tensors='pt').input_ids
        output_string = self.tokenizer.decode(
            decode_tokens[0], skip_special_tokens=True, max_new_tokens=self.model.config.max_length
        )
        return output_string.lstrip()  # Remove trailing space from tokenization

    def __call__(self, input_txt: str, mode: Optional[Literal['word_list', 'highest_prob', 'classify']] = None,
                 tokenizer_kwargs: Optional[dict] = None, generate_kwargs: Optional[dict] = None,
                 n_perturbations: int = 1, early_stopping: bool = False) -> Union[str, ValueError]:
        """
        Perturbs the input text using the specified mode and returns the perturbed text. No target word or attribute
        needs to be specified for this method.

        Args:
            input_txt: The input text to be perturbed

            mode: The mode to be used for perturbation

            tokenizer_kwargs: Additional keyword arguments to be passed to the tokenizer

            generate_kwargs: Additional keyword arguments to be passed to the generate method

            n_perturbations: The number of perturbations to be generated

            early_stopping: If True, the first perturbation that differs from the input is returned.

        Returns:
            Perturbed version of the input text
        """
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if generate_kwargs is None:
            generate_kwargs = {}
        if n_perturbations < 1:
            raise ValueError(f"At least one perturbation needs to be tried. Received {n_perturbations} as argument.")
        if mode is not None and mode not in {'word_list', 'classify'}:  # TODO rename these
            raise ValueError(f"Mode {mode} is invalid. Please choose from 'highest_prob, 'word_list' or 'classify'.")
        if mode is None:
            mode = 'word_list' if self.config.conditional else 'classify'

        if mode == 'word_list':
            targets = [w for w in input_txt.split(" ") if w in self.panda_dict]
            perturbations = [(t, perturbed) for t in targets for perturbed in self.panda_dict[t]]
        elif mode == 'classify':
            perturbations = self.generate_conditions(input_txt, n_perturbations, tokenizer_kwargs)
        random.shuffle(perturbations)
        perturbations = perturbations[:n_perturbations]
        texts, probabilities = [], []

        for word, attribute in perturbations:
            generated_txt, probability = self.generate(input_txt, word=word, attribute=attribute,
                                                       tokenizer_kwargs=tokenizer_kwargs,
                                                       generate_kwargs=generate_kwargs)
            if generated_txt != input_txt:
                if early_stopping:
                    return generated_txt
                else:
                    texts.append(generated_txt)
                    probabilities.append(probability)

        return texts[np.argmax(probabilities)] if texts else input_txt


class PerturberTemplate:
    """
    A template for generating perturbed text from input text, word and attribute. A distinction is made between the
    template of the original paper and the template used to train models with this library, which use the <SEP> token
    instead of a comma. Words are prefixed with a space so that the target word and attribute have the same token as
    their occurrences in the input text.
    """

    def __init__(self, sep: str = ",", pert_sep: str = "<PERT_SEP>",
                 original: bool = False, conditional: bool = True) -> None:
        self.sep = sep
        self.conditional = conditional
        self.pert_sep = pert_sep if not original else f" {pert_sep}"

    def __call__(self, input_txt: str, word: str = "", attribute: str = "") -> str:
        if not self.conditional:
            return f"{attribute_to_token(attribute)} {word}{self.pert_sep} {input_txt}"
        else:
            return f"{word}{self.sep} {attribute}{self.pert_sep} {input_txt}"

    def get_sentence_prefix(self, word: str = "", attribute: str = "") -> Union[str, ValueError]:
        if not self.conditional:
            return f"{attribute_to_token(attribute)} {word}{self.pert_sep}"
        else:
            return ValueError("Sentence prefix not available for conditional perturber")

    def get_attribute_prefix(self, word: str = "", attribute: str = "") -> Union[str, ValueError]:
        if not self.conditional:
            return ""
        else:
            return ValueError("Attribute prefix not available for conditional perturber")

    def get_word_prefix(self, word: str = "", attribute: str = "") -> Union[str, ValueError]:
        if not self.conditional:
            return f"{attribute_to_token(attribute)}"
        else:
            return ValueError("Word prefix not available for conditional perturber")
