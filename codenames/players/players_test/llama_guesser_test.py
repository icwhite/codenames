import unittest
from codenames.players.llama_guesser import LlamaGuesser

class TestLlamaGuesser(unittest.TestCase):
    def test_guess_make_guess(self):
        unselected = ["fall", "wind", "luck", "drop", "light", "ghost", "receipt", "check", "play"]
        clue = "weather"
        guesser = LlamaGuesser()
        
        single_guess = guesser.make_guess(
            unselected=unselected,
            clue=clue,
            num_targets=1
        )
        
        # For num_targets=1, we should get one guess
        assert len(single_guess) == 1

        multi_guesses = guesser.make_guess(
            unselected=unselected,
            clue=clue,
            num_targets=2
        )

        # For num_targets=2, we should get two guesses
        assert len(multi_guesses) == 2

        # Deterministically, we should get the same guess for the first word
        assert multi_guesses[0]  == single_guess[0]

        # All guesses should be in `unselected`
        for g in multi_guesses:
            assert g in unselected

    def test_guess_probabilities(self):
        unselected = ["fall", "wind", "luck", "drop", "light", "ghost", "receipt", "check", "play"]
        clue = "weather"
        guesser = LlamaGuesser()

        probs = guesser.guess_probabilities(
            unselected=unselected,
            clue=clue)
        
        # There should be a probability for each word
        assert len(probs) == len(unselected)
        

if __name__ == '__main__':
    unittest.main()