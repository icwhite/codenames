import json
from typing import List, Optional, Sequence, Tuple

from codenames.board import Board
from codenames.players import BaseGiver, BaseGuesser
from codenames.game import Game

class BatchedGame(Game):
    def __init__(
        self, 
        board: Board,
        giver: BaseGiver,
        guesser: BaseGuesser,
        verbose: bool = True,
        clue_list: Optional[str]='assets/clue_list.json'
    ):
        super().__init__(board, giver, guesser, verbose, clue_list)
    