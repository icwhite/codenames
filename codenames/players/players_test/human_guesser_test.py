from contextlib import AbstractContextManager
from typing import Any
import unittest
from codenames.players.human_guesser import HumanGuesser

class TestLiteralGuesser(unittest.TestCase):

    def test_guess(self):
        unselected = ["fall", "wind", "luck", "drop", "light", "ghost", "receipt", "check", "play"]
        clue = "weather"
        guesser = HumanGuesser()
        guess = guesser.make_guess(unselected, clue)
        print(guess)
        guess = guesser.make_guess(unselected, clue, num_targets=2)

if __name__ == '__main__':
    unittest.main()