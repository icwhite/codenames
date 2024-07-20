import json
from typing import List, Optional, Sequence, Tuple

from codenames.board import Board
from codenames.players import BaseGiver, BaseGuesser
import os

class Game():

    def __init__(
        self, 
        board: Board,
        giver: BaseGiver,
        guesser: BaseGuesser,
        verbose: bool = True,
        clue_list: Optional[str]='assets/clue_list.json'
    ):
        self.board = board
        self.giver = giver
        self.guesser = guesser
        self.verbose = verbose
        self.status = "running"

        clues = set(json.load(open(clue_list)))
        self.clue_list = list(clues - self.board.all_words)
        for word in self.board.all_words:
            for clue in self.clue_list:
                if word in clue:
                    self.clue_list.remove(clue)
                if clue in word:
                    self.clue_list.remove(clue)
    

    def play_giver(self, 
                   targets_options: Optional[dict]={},
                   clues_options: Optional[dict]={}) -> Tuple[str, Sequence[str]]:
        if self.verbose:
            print(f"Remaining goal words: {', '.join(self.board.unselected_goal_words)}")
            print(f"Remaining avoid words: {', '.join(self.board.unselected_avoid_words)}")
            print(f"Remaining neutral words: {', '.join(self.board.unselected_neutral_words)}\n")
            print("CLUE GIVER'S TURN")
            
        targets = self.giver.select_targets(
            goal=self.board.unselected_goal_words,
            avoid=self.board.unselected_avoid_words,
            neutral=self.board.unselected_neutral_words,
            clues=self.clue_list,
        )
        clue = self.giver.give_clue(
            goal=self.board.unselected_goal_words,
            avoid=self.board.unselected_avoid_words,
            neutral=self.board.unselected_neutral_words,
            clues=self.clue_list,
            targets=targets,
        )
        if self.verbose:
            print(f"Targets selected: {', '.join(targets)}")
            print(f"Clue: {clue}\n")
        return clue, targets
    
    
    def play_guesser(
        self,
        clue: str,
        targets: Sequence[str],
        options: Optional[dict]={},
    ) -> Sequence[str]:
        if self.verbose:
            print("GUESSER'S TURN")
        guess = self.guesser.make_guess(
            unselected=self.board.unselected_words,
            clue=clue,
            num_targets=len(targets),
            **options,
        )
        if self.verbose:
            print(f"Guessed words: {', '.join(guess)}")

        giver_logs = self.giver.observe_turn(
            goal=self.board.unselected_goal_words,
            avoid=self.board.unselected_avoid_words,
            neutral=self.board.unselected_neutral_words,
            targets=targets,
            clue=clue,
            guess=guess,
            not_guess=self.board.get_not_guess(guess),
        )
            
        # copy the previously unselected words before we make selections from this round
        unselected = self.board.unselected_words.copy()
        result = [self.board.select_word(g) for g in guess]

        if self.verbose:
            print(f"Result: {', '.join(result)}")
            print("====================================================================================\n")

        if "avoid" in result:
            self.status = "loss"
            if self.verbose:
                print("GAME OVER: an avoid word has been selected")
        elif len(self.board.unselected_goal_words) == 0:
            self.status = "win"
            if self.verbose:
                print("GAME WON: all goal words have been selected!")

        return result
    
class TrainingGame(Game):
    def __init__(
        self, 
        board: Board,
        giver: BaseGiver,
        guesser: BaseGuesser,
        verbose: bool = True,
        clue_list: Optional[str]='assets/clue_list.json'
    ):
        super().__init__(board, giver, guesser, verbose, clue_list)
    
    def play_guesser(
        self,
        clue: str,
        targets: Sequence[str],
        options: Optional[dict]={},
    ) -> Sequence[str]:
        if self.verbose:
            print("GUESSER'S TURN")
        guess = self.guesser.make_guess(
            unselected=self.board.unselected_words,
            clue=clue,
            num_targets=len(targets),
            **options,
        )
        if self.verbose:
            print(f"Guessed words: {', '.join(guess)}")
            
        giver_logs = self.giver.observe_turn(
            goal=self.board.unselected_goal_words,
            avoid=self.board.unselected_avoid_words,
            neutral=self.board.unselected_neutral_words,
            targets=targets,
            clue=clue,
            guess=guess,
            not_guess=self.board.get_not_guess(guess),
        )
        
        # copy the previously unselected words before we make selections from this round
        unselected = self.board.unselected_words.copy()
        result = [self.board.select_word(g) for g in guess]
        
        guesser_logs = self.guesser.observe_turn(
            unselected=unselected,
            clue=clue,
            num_targets=len(targets),
            guess=guess,
            result=result,
            **options,
        )

        if self.verbose:
            print(f"Result: {', '.join(result)}")
            print("====================================================================================\n")

        if "avoid" in result:
            self.status = "loss"
            if self.verbose:
                print("GAME OVER: an avoid word has been selected")
        elif len(self.board.unselected_goal_words) == 0:
            self.status = "win"
            if self.verbose:
                print("GAME WON: all goal words have been selected!")

        return result, giver_logs, guesser_logs