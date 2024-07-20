import numpy as np
from typing import Optional, Sequence

from codenames.players import BaseGuesser
import random
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity


class HumanGuesser(BaseGuesser):

    def __init__(self,):
        super().__init__()

    def make_guess(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int = 1,
        choose_argmax: Optional[bool] = False,
    ) -> Sequence[str]:
        print("="*100)
        print("clue: ", clue)
        print("="*100)
        print(f"select {num_targets} word(s) from the following list:\n {unselected}")
        selected_guesses = []
        for i in range(num_targets):
            selected_guess = ""
            while selected_guess not in unselected or selected_guess in selected_guesses:
                selected_guess = input(f"Enter guess number {i+1}: ").lower().strip(" ")
                if selected_guess not in unselected:
                    print(f"Invalid guess. Please select from the following list: {unselected}")
                if selected_guess in selected_guesses:
                    print(f"Word already selected. Please select a different word.")
            selected_guesses.append(selected_guess)
        return selected_guesses

    def observe_turn(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
        guess: Sequence[str],
        result: Sequence[str],
    ):
        return {}

    def guess_probabilities(
        self,
        unselected: Sequence[str],
        clue: str,
    ) -> Sequence[float]:
        return np.ones(len(unselected)) / len(unselected)
        