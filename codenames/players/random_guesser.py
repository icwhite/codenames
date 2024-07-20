import numpy as np
from typing import Sequence

from codenames.players import BaseGuesser


class RandomGuesser(BaseGuesser):

    def __init__(self):
        super().__init__()


    def make_guess(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
    ) -> Sequence[str]:
        return np.random.choice(unselected, num_targets, replace=False)


    def observe_turn(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
        guess: Sequence[str],
        result: Sequence[str],
    ):
        pass

    def guess_probabilities(self, 
                            unselected: Sequence[str],
                            clue: str) -> Sequence[float]:
        return np.random.rand(len(unselected))
        