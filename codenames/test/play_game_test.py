import numpy as np
from codenames.board import Board
from codenames.game import Game
from codenames.players import RandomGiver, RandomGuesser
import os
import unittest

class TestLiteralGuesser(unittest.TestCase):
    def test_play_game(self):
        board = Board(cultural_codes_list='assets/cultural_codes_list.json')
        giver = RandomGiver()
        guesser = RandomGuesser()
        game = Game(board, giver, guesser, clue_list="assets/clue_list.json")
        while game.status == "running":
            clue, targets = game.play_giver()
            results = game.play_guesser(clue, targets)
    
    def test_win_game(self):
        board = Board(num_words=9, num_goal=9, num_avoid=0, cultural_codes_list='assets/cultural_codes_list.json')
        giver = RandomGiver()
        guesser = RandomGuesser()
        game = Game(board, giver, guesser, clue_list="assets/clue_list.json")
        while game.status == "running":
            clue, targets = game.play_giver()
            results = game.play_guesser(clue, targets)
        self.assertEqual(game.status, "win")
    
    def test_lose_game(self):
        board = Board(num_words=9, num_goal=0, num_avoid=9, cultural_codes_list='assets/cultural_codes_list.json')
        giver = RandomGiver()
        guesser = RandomGuesser()
        game = Game(board, giver, guesser, clue_list="assets/clue_list.json")
        while game.status == "running":
            clue, targets = game.play_giver()
            results = game.play_guesser(clue, targets)
        self.assertEqual(game.status, "win")

if __name__ == '__main__':
    unittest.main()