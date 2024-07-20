import numpy as np
from typing import Sequence

from codenames.players import BaseGiver


class RandomGiver(BaseGiver):

    def __init__(self):
        super().__init__()


    def select_targets(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: Sequence[str],
    ) -> Sequence[str]:
        return np.random.choice(goal, min(2, len(goal)), replace=False)


    def give_clue(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clues: Sequence[str],
    ) -> str:
        return "clue"


    def observe_turn(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
        not_guess: Sequence[str],
    ):
        pass
