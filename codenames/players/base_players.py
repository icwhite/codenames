from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence


class BaseGiver(ABC):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def select_targets(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: Sequence[str]=None,
    ) -> Sequence[str]:
        ...


    @abstractmethod
    def give_clue(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clues: Sequence[str]=None,
    ) -> str:
        ...


    @abstractmethod
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
        ...


class BaseGuesser(ABC):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def make_guess(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
    ) -> Sequence[str]:
        ...


    @abstractmethod
    def observe_turn(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
        guess: Sequence[str],
        result: Sequence[str],
    ):
        ...
    
    @abstractmethod
    def guess_probabilities(
        self,
        unselected: Sequence[str],
        clue: str,
    ) -> Sequence[float]:
        ...
        