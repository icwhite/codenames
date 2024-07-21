Code for the paper "Communicate to Play: Pragmatic Reasoning for Efficient Cross-Cultural Communication in Codenames" by Isadora White, Sashrika Pandey, and Michelle Pan.

We provide our code for:
1. Creating Codenames players using contrastive learning of an embedding space and LLM prompting
2. Studying culturally induced differences in common ground in trained models
3. Implementing RSA+C3 to infer socioculutral context in interaction

# Installation
Run ```pip install -e .```

# Experiments

We utilize the [Cultural Codes Dataset](https://github.com/SALT-NLP/codenames) for running our Codenames Duet experiment. This repository should be cloned in the root directory.

## Word Embeddings

## LLM Prompting
Experiment files for clue selection and guess generation using the cross cultural codes dataset can be found in `llama_exps/`. You will need to download weights for Llama2; instructions can be found in the [Llama repo](https://github.com/meta-llama/llama). To run experiments with the Llama models, this repository should be cloned in the root directory.

## Interaction

# Citing
Coming soon!
