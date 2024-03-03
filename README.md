# Perturbers

This codebase is built upon the great work of [Qian et. al. (2022)](https://arxiv.org/abs/2205.12586). Using this
library, you can easily integrate neural augmentation models with your NLP pipelines or train new perturber models from
scratch.

# Installation

`perturbers` is available on PyPI and can be installed using pip:

```bash
pip install perturbers
```

# Usage

Using a perturber is as simple as creating a new instance of the `Perturber` class and calling the `generate` method
with the sentence you want to perturb along with the target word and the attribute you want to change:

```python
from perturbers import Perturber

perturber = Perturber()
unperturbed = "Jack was passionate about rock climbing and his love for the sport was infectious to all men around him."
perturber.generate(unperturbed, "Jack", "female")
# "Jane was passionate about rock climbing and her love for the sport was infectious to all men around her."
```

You can also perturb a sentence without specifying a target word or attribute:

```python
perturber("Jack was passionate.", retry_unchanged=True)
# "Jackie was passionate."
```

## Training a new perturber model

To train a new perturber model, take a look at the `train_perturber.py` script. This script will train a new perturber
model using the PANDA dataset. Currently the scripts only support training BART models, but any encoder-decoder model
can be used.

Perturber models are evaluated based on the following metrics:

- `bleu4`: The 4-gram BLEU score of the perturbed sentence compared to the original sentence
- `perplexity`: The perplexity of the perturbed sentence
- `perplexity_perturbed`: The perplexity of only the perturbed tokens from the perturbed sentence

## Pre-trained models

In addition to the codebase, we also provide pre-trained perturber models in a variety of sizes:

|                                                                   | Base model                                                   | Parameters | Perplexity | Perplexity (perturbed idx)* | BLEU4  |
|-------------------------------------------------------------------|--------------------------------------------------------------|------------|------------|-----------------------------|--------|
| [perturber-small](https://huggingface.co/fairnlp/perturber-small) | [bart-small](https://huggingface.co/lucadiliello/bart-small) | 70M        | 1.076      | 4.079                       | 0.822  |
| [perturber-base](https://huggingface.co/fairnlp/perturber-base)   | [bart-base](https://huggingface.co/facebook/bart-base)       | 139M       | 1.058      | 2.769                       | 0.794  |
| [perturber (original)](https://huggingface.co/facebook/perturber) | [bart-large](https://huggingface.co/facebook/bart-large)     | 406M       | 1.06**     | N/A                         | 0.88** |

*Measures perplexity only of perturbed tokens, as the majority of tokens remains unchanged leading to Perplexity scores
approaching 1

**The perplexity and BLEU4 scores are those reported in the original paper and not measured via this codebase.

# Roadmap

- [x] Add default perturber model
- [x] Pretrain small and medium perturber models
- [ ] Train model to identify target words and attributes
- [ ] Add training of unconditional perturber models (i.e. only get a sentence, no target word/attribute)
- [ ] Add self-training by pretraining perturber base model (e.g. BART) on self-perturbed data

Other features could include

- [ ] Data cleaning of PANDA (remove non-target perturbations)
- [ ] Multilingual perturbation

# Read more

- [Original perturber paper](https://aclanthology.org/2022.emnlp-main.646/)