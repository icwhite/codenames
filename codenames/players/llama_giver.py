import numpy as np
from typing import Sequence

from codenames.players import BaseGiver
from llama_exps.utils import load_model_and_tokenizer, get_fixed_generation_prob, hf_llama_generate

class LlamaGiver(BaseGiver):
    def __init__(self, model_name):
        super().__init__()
        model, tokenizer = load_model_and_tokenizer(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = 2 if 'llama3' in model_name else 4 # set llama-3 batch size to be lower

    def select_targets(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
    ) -> Sequence[str]:
        joined_avoid = ", ".join(avoid)
        joined_neutral = ", ".join(neutral)
        joined_goal = ", ".join(goal)
        prompt = f"You are playing Codenames and are the clue giver. Words to avoid are {joined_avoid}. Neutral words are {joined_neutral}. All words on the board are {joined_goal}. The word you would select for your teammate to guess is "

        probs = get_fixed_generation_prob(self.model, 
                                          self.tokenizer, 
                                          [prompt] * len(goal), 
                                          goal, 
                                          self.batch_size)

        # output the two most likely targets
        zipped = [item for item in zip(goal, probs)]
        zipped.sort(key = lambda x: x[1], reverse=True)
        assert zipped[0][1] > zipped[1][1]
        return [i[0] for i in zipped[:2]]


    def give_clue(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
    ) -> str:
        joined_avoid = ", ".join(avoid)
        joined_neutral = ", ".join(neutral)
        joined_targets = ", ".join(targets)
        joined_goal = ", ".join(goal)
        prompt = f"You are playing Codenames and are the clue giver. Words to avoid are {joined_avoid}. Neutral words are {joined_neutral}. Out of the words {joined_goal}, your goal is to get your teammate to guess the words {joined_targets} using a phrase of words. A clue to give your teammate is "
        
        gens, _ = hf_llama_generate(self.model, 
                                self.tokenizer, 
                                prompts=[prompt], 
                                max_new_tokens=3, 
                                batch_size=self.batch_size)
        assert len(gens) == 1
        return gens[0]


    def observe_turn(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
    ):
        pass
