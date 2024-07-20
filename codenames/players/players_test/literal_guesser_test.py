from contextlib import AbstractContextManager
from typing import Any
import unittest
from codenames.players.literal_guesser import LiteralGuesser
from codenames.embeddings import BertEmbeddings, GloveEmbeddings
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration

class TestLiteralGuesser(unittest.TestCase):
    def test_guess_single_token(self):
        unselected = ["fall", "wind", "luck", "drop", "light", "ghost", "receipt", "check", "play"]
        clue = "weather"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = BertEmbeddings(tokenizer, model)
        guesser = LiteralGuesser(embeddings)
        guess = guesser.make_guess(unselected, clue)
        print(guess)
        argmax_guess = guesser.make_guess(unselected, clue, choose_argmax=True)
        print(argmax_guess)

    def test_guess_multiple_tokens(self):
        unselected = ["fall more token", "wind", "luck", "drop more token", "light", "ghost", "receipt", "check", "play"]
        clue = "weather weather"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        embeddings = BertEmbeddings(tokenizer, model)
        guesser = LiteralGuesser(embeddings)
        guess = guesser.make_guess(unselected, clue)
    
    def test_guess_glove_embeddings(self):
        unselected = ["fall", "wind", "luck", "drop", "light", "ghost", "receipt", "check", "play"]
        clue = "weather"
        embeddings = GloveEmbeddings()
        guesser = LiteralGuesser(embeddings)
        guess = guesser.make_guess(unselected, clue)
        print(guess)
        argmax_guess = guesser.make_guess(unselected, clue, choose_argmax=True)
        print(argmax_guess)

if __name__ == '__main__':
    unittest.main()