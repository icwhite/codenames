import unittest
from codenames.players.llama_giver import LlamaGiver

class TestPragmaticGiver(unittest.TestCase):
    def test_select_targets(self):
        goal = ["fall", "wind", "luck", "drop"]
        avoid = ["receipt", "check"]
        neutral = ["light", "ghost", "play"]
        giver = LlamaGiver()

        target = giver.select_targets(
            goal=goal,
            avoid=avoid,
            neutral=neutral,
        ) 
        
        # the giver should output two target words
        print("target: ", target)
        assert len(target) == 2
        for t in target:
            assert t in goal

    def test_give_clue(self):
        goal = ["fall", "wind", "luck", "drop"]
        avoid = ["receipt", "check"]
        neutral = ["light", "ghost", "play"]
        giver = LlamaGiver()

        clue_1 = giver.give_clue(
            goal=goal,
            avoid=avoid,
            neutral=neutral,
            targets=["fall"]
        )

        # check that we output a clue for every list of targets
        assert isinstance(clue_1, str)
        print(clue_1)

        clue_2 = giver.give_clue(
            goal=goal,
            avoid=avoid,
            neutral=neutral,
            targets=["fall", "luck"]
        )
        assert isinstance(clue_2, str)
        print(clue_2)

        # the clues are likely different 
        assert clue_1 != clue_2
    
if __name__ == '__main__':
    unittest.main()