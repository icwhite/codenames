import time

from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration

from codenames.board import Board
from codenames.embeddings import BertEmbeddings
from codenames.game import Game
from codenames.players import LiteralGuesser, PragmaticGiver


# MAKE GAME ELEMENTS
board = Board()
board.set_words(
    goal_words=["fall", "wind", "luck", "drop"],
    avoid_words=["receipt", "check"],
    neutral_words=["light", "ghost", "play"],
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
embeddings = BertEmbeddings(tokenizer, model)
guesser = LiteralGuesser(embeddings, choose_argmax=True)
pragmatic_giver = PragmaticGiver(guesser, embeddings)
game = Game(board, pragmatic_giver, guesser)

# GIVER: SELECT TARGET
start = time.time()
targets = pragmatic_giver.select_targets(
    goal=board.unselected_goal_words,
    avoid=board.unselected_avoid_words,
    neutral=board.unselected_neutral_words,
    clues=game.clue_list,
) # 4.9 seconds
print(f"targets: {targets}")
target_time = time.time() - start

# GIVER: GIVE CLUE
clue = pragmatic_giver.give_clue(
    goal=board.unselected_goal_words,
    avoid=board.unselected_avoid_words,
    neutral=board.unselected_neutral_words,
    clues=game.clue_list,
    targets=targets,
) # 4.5 seconds
print(f"clue: {clue}")
clue_time = time.time() - target_time - start
print(f"target time: {target_time}, clue_time: {clue_time}")

# GUESSER: MAKE GUESS
guess = guesser.make_guess(
    unselected=board.unselected_words,
    clue=clue,
)
print(f"guess: {guess}")
guess_time = time.time() - clue_time - target_time - start
print(f"guess time: {guess_time}")