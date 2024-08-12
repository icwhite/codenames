Code for the paper [Communicate to Play: Pragmatic Reasoning for Efficient Cross-Cultural Communication in Codenames](https://arxiv.org/abs/2408.04900) by Isadora White, Sashrika Pandey, and Michelle Pan.

![rsac3](https://github.com/user-attachments/assets/38314436-d110-4021-bdc9-1a30a0923014)

We provide our code for:
1. Creating Codenames players using contrastive learning of an embedding space and LLM prompting
2. Studying culturally induced differences in common ground in trained models
3. Implementing RSA+C3 to infer socioculutral context in interaction

# Abstract
Cultural differences in common ground may result in pragmatic failure and misunderstandings during communication. We develop our method Rational Speech Acts for Cross-Cultural Communication (RSA+C3) to resolve cross-cultural differences in common ground. To measure the success of our method, we study RSA+C3 in the collaborative referential game of Codenames Duet and show that our method successfully improves collaboration between simulated players of different cultures. Our contributions are threefold: (1) creating Codenames players using contrastive learning of an embedding space and LLM prompting that are aligned with human patterns of play, (2) studying culturally induced differences in common ground reflected in our trained models, and (3) demonstrating that our method RSA+C3 can ease cross-cultural communication in gameplay by inferring sociocultural context from interaction.

# Installation
Run ```pip install -e .```

# Experiments

We utilize the [Cultural Codes Dataset](https://github.com/SALT-NLP/codenames) for running our Codenames Duet experiment. This repository should be cloned in the root directory.

## Word Embeddings
You can find code for word embeddings in codenames/embeddings and code for training embeddings in `codenames/embeddings/glove_embeddings.py`. You can find code for evaluating WE and random givers and guessers in `codenames/codenames/eval/`.

## LLM Prompting
Experiment files for clue selection and guess generation using the cross cultural codes dataset can be found in `llama_exps/`. You will need to download weights for Llama2; instructions can be found in the [Llama repo](https://github.com/meta-llama/llama). To run experiments with the Llama models, this repository should be cloned in the root directory.

## Interaction
Files for RSA+C3 codenames giver can be found in `codenames/players/match_giver.py`, RSA in `codenames/players/pragmatic_giver.py` and the literal giver in `codenames/players/similarities_giver.py`. To run interaction experiments you can run these in `codenames/eval/interactive_evaluation.py`. 

# Citing
If you use our code, please consider citing:
```
@misc{white2024communicatetoplay,
      title={Communicate to Play: Pragmatic Reasoning for Efficient Cross-Cultural Communication in Codenames}, 
      author={Isadora White and Sashrika Pandey and Michelle Pan},
      year={2024},
      eprint={2408.04900},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04900}, 
}
```
