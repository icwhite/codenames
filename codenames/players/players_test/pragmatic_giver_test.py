from contextlib import AbstractContextManager
from typing import Any
import unittest
from codenames.embeddings import BertEmbeddings, GloveEmbeddings, TrainGuesserEmbeddings
from codenames.players.literal_guesser import LiteralGuesser
# from codenames.players.similarities_giver import PragmaticGiver
from codenames.players.similarities_giver import SimilaritiesGiver
from codenames.eval import interactive_evaluation
from transformers import BertTokenizer, BertModel

import json 

class TestSimilaritiesGiver(unittest.TestCase):
    
    def test_select_target_single_token(self):
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = GloveEmbeddings()
        guesser = LiteralGuesser(embeddings)
        clue_list = ["slip", "dog", "season", "weather", "annoyinglyyyyy"]
        similarities_giver = SimilaritiesGiver(embeddings)
        targets = similarities_giver.select_targets(goal=["fall", "wind", "luck", "drop"],
                                         avoid=["receipt", "check"], 
                                         neutral=["light", "ghost", "play"],
                                         clues=clue_list)
        more_targets = similarities_giver.select_targets(goal=["fall", "wind", "luck", "drop"],
                                            avoid=["receipt", "check"],
                                            neutral=["light", "ghost", "play"],
                                            clues=clue_list, num_targets=3)
        print(targets)
        print(more_targets)
    
    def test_select_target_train_embeddings(self):
        embeddings = TrainGuesserEmbeddings()
        guesser = LiteralGuesser(embeddings)
        clue_list = ["slip", "dog", "season", "weather", "annoyinglyyyyy"]
        similarities_giver = SimilaritiesGiver(embeddings)
        targets = similarities_giver.select_targets(goal=["fall", "wind", "luck", "drop"],
                                         avoid=["receipt", "check"], 
                                         neutral=["light", "ghost", "play"],
                                         clues=clue_list)
        more_targets = similarities_giver.select_targets(goal=["fall", "wind", "luck", "drop"],
                                            avoid=["receipt", "check"],
                                            neutral=["light", "ghost", "play"],
                                            clues=clue_list, num_targets=3)
        print(targets)
        print(more_targets)
    
    def test_give_clue_glove_embeddings(self):
        embeddings = GloveEmbeddings()
        guesser = LiteralGuesser(embeddings)
        clue_list = ["slip", "dog", "season", "weather", "annoyingly"]
        similarities_giver = SimilaritiesGiver(embeddings)
        targets = similarities_giver.select_targets(goal=["fall", "wind", "luck", "drop"],
                                         avoid=["receipt", "check"], 
                                         neutral=["light", "ghost", "play"],
                                         clues=clue_list)
        clue = similarities_giver.give_clue(goal=["fall", "wind", "luck", "drop"],
                                            avoid=["receipt", "check"], 
                                            neutral=["light", "ghost", "play"],
                                            clues=clue_list, 
                                            targets=targets)
        argmax_clue = similarities_giver.give_clue(goal=["fall", "wind", "luck", "drop"],
                                            avoid=["receipt", "check"], 
                                            neutral=["light", "ghost", "play"],
                                            clues=clue_list, 
                                            targets=targets, choose_argmax=True)
        print("glove")
        print(targets, clue)

    def test_pragmatic_giving(self):

        """
        A scenario that was not working in interaction: 

        Remaining goal words: lead, stock, spring, draft, spy, cover, king, row, boom
        Remaining avoid words: bond, center, crash
        Remaining neutral words: contract, cold, centaur, soul, march, space, comic,
        strike, revolution, degree, leprechaun, force, part
        """
        goal = ["lead", "stock", "spring", "draft", "spy", "cover", "king", "row", "boom"]
        avoid = ["bond", "center", "crash"]
        neutral = ["contract", "cold", "centaur", "soul", "march", "space", "comic",
                    "strike", "revolution", "degree", "leprechaun", "force", "part"]
        
        all_words = set(goal + avoid + neutral)
        clue_list = 'assets/clue_list.json'
        with open(clue_list, "r") as f:
            clues = set(json.load(f))
        clue_list = list(clues - all_words)
        embeddings = GloveEmbeddings()
        guesser = LiteralGuesser(embeddings, choose_argmax=True)
        similarities_giver = SimilaritiesGiver(embeddings, choose_argmax=True)
        targets = similarities_giver.select_targets(goal=goal,
                                            avoid=avoid, 
                                            neutral=neutral,
                                            clues=clue_list, 
                                            num_targets=3)
        print("="*100)
        print("Test Case 4")
        print("best target from test case 4", targets)
        clue = similarities_giver.give_clue(goal=goal,
                                            avoid=avoid, 
                                            neutral=neutral,
                                            clues=clue_list, 
                                            targets=targets, 
                                            choose_argmax=True)
        print("clue from test case 4", clue)
        guess = guesser.make_guess(unselected=list(all_words), clue=clue, num_targets=len(targets), choose_argmax=True)
        print("guess from test case 4", guess)
        # assert (set(guess) == set(targets)), f"expected {targets}, got {guess}"
    
    def test_avoid_word_chosen(self):
        """
        Remaining goal words: ninja, press
        Remaining avoid words: genius, state, leprechaun
        Remaining neutral words: cycle, trip, poison, check, beat, alien, witch, sub

        """
        goal = ["ninja", "press"]
        avoid = ["genius", "state", "leprechaun"]
        neutral = ["cycle", "trip", "poison", "check", "beat", "alien", "witch", "sub"]

        all_words = set(goal + avoid + neutral)
        clue_list = 'assets/clue_list.json'
        with open(clue_list, "r") as f:
            clues = set(json.load(f))
        clue_list = list(clues - all_words)

        embeddings = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )

        embeddings.load_weights("models/education_high_school_associate_300_1000.pth")


        guesser = LiteralGuesser(embeddings, choose_argmax=True)
        similarities_giver = SimilaritiesGiver(embeddings, choose_argmax=True, adaptive=False)

        targets = similarities_giver.select_targets(goal=goal,
                                            avoid=avoid, 
                                            neutral=neutral,
                                            clues=clue_list, 
                                            num_targets=2)
        print("="*100)
        print("Test Case 5")
        print("best target from test case 5", targets)
        clue = similarities_giver.give_clue(goal=goal,
                                            avoid=avoid, 
                                            neutral=neutral,
                                            clues=clue_list, 
                                            targets=targets, 
                                            choose_argmax=True)
        print("clue from test case 5", clue)
        guess = guesser.make_guess(unselected=list(all_words), clue=clue, num_targets=len(targets), choose_argmax=True)
        print("guess from test case 5", guess)
        # assert (set(guess) == set(targets)), f"expected {targets}, got {guess}"

    
    # def test_play_game_eval(self):
    #     embeddings = GloveEmbeddings()
    #     guesser = LiteralGuesser(embeddings, choose_argmax=True)
    #     similarities_giver = PragmaticGiver(guesser, embeddings, choose_argmax=True)
    #     pass

if __name__ == '__main__':
    unittest.main()