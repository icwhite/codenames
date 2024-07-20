import json
import numpy as np
from typing import Optional, Sequence

class Board():

    def __init__(
        self, 
        num_words: Optional[int] = 25,
        num_goal: Optional[int] = 9,
        num_avoid: Optional[int] = 3,
        dictionary: Optional[Sequence[str]] = None,
        seed: Optional[int] = 0,
        cultural_codes_list: Optional[str] = 'assets/cultural_codes_list.json',
    ):
        np.random.seed(seed)

        self.num_words = num_words
        self.num_goal = num_goal
        self.num_avoid = num_avoid
        self.dictionary = dictionary

        if dictionary is None:
            self.dictionary = json.load(open(cultural_codes_list))
        else:
            self.dictionary = dictionary

        self.reset_board()

    def reset_board(self):
        indices = np.random.permutation(np.arange(len(self.dictionary)))
        words = [self.dictionary[i] for i in indices[:self.num_words]]
        goal_words = words[:self.num_goal]
        avoid_words = words[self.num_goal:self.num_goal+self.num_avoid]
        neutral_words = words[self.num_goal+self.num_avoid:]

        self.all_words = set(words)
        self.goal_words = set(goal_words)
        self.avoid_words = set(avoid_words)
        self.neutral_words = set(neutral_words)
        self.selected_words = set()
        

    def set_words(
        self,
        goal_words: Sequence[str],
        avoid_words: Sequence[str],
        neutral_words: Sequence[str],
    ):
        self.goal_words = set(goal_words)
        self.avoid_words = set(avoid_words)
        self.neutral_words = set(neutral_words)
        self.all_words = self.goal_words ^ self.avoid_words ^ self.neutral_words

        assert (
            (not self.goal_words & self.avoid_words) and
            (not self.avoid_words & self.neutral_words) and
            (not self.neutral_words & self.goal_words)
        ), "goal, avoid, and neutral words cannot overlap"


    def select_word(
        self,
        word: str,
    ) -> str:
        assert word in self.all_words, f"word {word} is not on the board"
        assert word not in self.selected_words, f"word {word} has already been selected"

        self.selected_words.add(word)
        if word in self.goal_words:
            return "goal"
        elif word in self.neutral_words:
            return "neutral"
        elif word in self.avoid_words:
            return "avoid"
        else:
            raise NotImplementedError
        
    def get_not_guess(
        self,
        guess: Sequence[str],
    ) -> Sequence[str]:
        return list(set(self.unselected_words) - set(guess))


    @property
    def unselected_words(self) -> Sequence[str]:
        return list(self.all_words - self.selected_words)
    
        
    @property
    def unselected_goal_words(self) -> Sequence[str]:
        return list(self.goal_words - self.selected_words)
    

    @property
    def unselected_avoid_words(self) -> Sequence[str]:
        return list(self.avoid_words - self.selected_words)
    
    
    @property
    def unselected_neutral_words(self) -> Sequence[str]:
        return list(self.neutral_words - self.selected_words)
    