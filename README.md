# Perturbers

This codebase is built upon the great work of [Qian et. al. (2022)](https://arxiv.org/abs/2205.12586). Using this
library, you can easily train and deploy perturbation augmentation models

# Roadmap

- [x] Add default perturber model
- [ ] Reproduce training of perturber model
- [ ] Pretrain small and medium perturber models
- [ ] Add training of unconditional perturber models (i.e. only get a sentence, no target word/attribute)
- [ ] Add self-training by pretraining perturber base model (e.g. BART) on self-perturbed data

Other features could include

- [ ] Data cleaning of PANDA (remove non-target perturbations)
- [ ] Multilingual perturbation
