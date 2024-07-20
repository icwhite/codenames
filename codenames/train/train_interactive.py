import numpy as np
import argparse
from codenames.embeddings import *
from codenames.board import Board
from codenames.embeddings.glove_embeddings import GloveEmbeddings
from codenames.game import Game, TrainingGame
from codenames.players import RandomGiver, RandomGuesser
from codenames.players.literal_guesser import LiteralGuesser
from codenames.players.llama_guesser import LlamaGuesser
from codenames.players.pragmatic_giver import PragmaticGiver
from codenames.players.selective_pragmatic_giver import SelectivePragmaticGiver 
from codenames.eval.interactive_evaluation import play_game_eval
import torch
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--desc", type=str, help="run description")
parser.add_argument("--train_set", type=str, default="train", help="split of data used for training")
parser.add_argument("--val_set", type=str, default="val", help="split of data used for val")
parser.add_argument("--glove_dim", type=int, choices={50, 100, 200, 300}, default=300, help="GloVe embedding dimension")
parser.add_argument("--embed_dim", type=int, default=1000, help="trained embedding dimension")
parser.add_argument("--loss", type=str, choices={"cosine_similarity", "l2", "clip"}, default="clip", help="loss function to use")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--verbose", action="store_true", help="print game state")
parser.add_argument("--giver_embeddings_path", type=str, default="", help="path to trained embeddings for clue giver")
parser.add_argument("--guesser_embeddings_path", type=str, default="", help="path to trained embeddings for guesser")
parser.add_argument("--llama-guesser", action=argparse.BooleanOptionalAction, help="use llama guesser")
parser.add_argument("--num_games", type=int, default=1000, help="number of games to play")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--log_every", type=int, default=25, help="log every n games")
parser.add_argument("--eval_every", type=int, default=100,  help="eval every n games")
args = parser.parse_args()
print(args)
wandb.init(
    project="codenames",
    name=f"interaction {args.desc}",
    config=vars(args),
)

print("Loading embeddings and dataset")
giver_embeddings = TrainGuesserEmbeddings(
    in_embed_dim=args.glove_dim,
    out_embed_dim=args.embed_dim,
    loss_fn=args.loss,
    lr=args.lr,
)

if args.giver_embeddings_path:
    giver_embeddings.load_weights(args.giver_embeddings_path)

if args.guesser_embeddings_path: 
    guesser_embeddings = TrainGuesserEmbeddings(
        in_embed_dim=args.glove_dim,
        out_embed_dim=args.embed_dim,
        loss_fn=args.loss,
        lr=args.lr,
    )
    guesser_embeddings.load_weights(args.guesser_embeddings_path)
else:
    guesser_embeddings = GloveEmbeddings(embed_dim=args.glove_dim)

# toggle which type of guesser to use
if args.llama_guesser:
    guesser = LlamaGuesser()
else:
    guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)

# load in literal guesser regardless for pragmatic giver
literal_guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)
giver = PragmaticGiver(literal_guesser, giver_embeddings, choose_argmax=False, k=20, adaptive=True)

num_won, num_targets, num_turns = 0, 0, 0
num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0
board = Board(seed=args.seed)
for i in range(args.num_games):
    board.reset_board()
    game = TrainingGame(board, 
                giver, 
                guesser, 
                verbose=args.verbose)
    
    while game.status == "running":
        clue, targets = game.play_giver()
        results, giver_logs, guesser_logs = game.play_guesser(clue, targets)
        if giver_logs["train_loss"]:
            print(giver_logs)
            wandb.log({"giver": giver_logs})
        for result in results:
            if result == "goal":
                num_goal_guessed += 1
            elif result == "neutral":
                num_neutral_guesed += 1
            elif result == "avoid":
                num_avoid_guessed += 1
        num_targets += len(targets)
        num_turns += 1
    if game.status == "win":
        num_won += 1
    if i % args.log_every == 0:
        print("training logs")
        train_logs = {
            "iter": i,
            "win_rate": num_won / args.log_every,
            "average_turns_per_game": num_turns / args.log_every,
            "average_targets_per_turn": num_targets / num_turns,
            "goal_guessed": num_goal_guessed / num_targets,
            "neutral_guessed": num_neutral_guesed / num_targets,
            "avoid_guessed": num_avoid_guessed / num_targets
        }
        print(train_logs)
        wandb.log({"train": train_logs})
        num_won, num_targets, num_turns = 0, 0, 0
        num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0
    
    if i % args.eval_every == 0:
        eval_giver = PragmaticGiver(guesser, giver_embeddings, choose_argmax=True, adaptive=False)
        assert giver.embeddings is eval_giver.embeddings
        print("="*100, "\n")
        print("eval")
        with torch.no_grad():
            logs = play_game_eval(guesser, eval_giver, num_games=100, seed=args.seed, verbose=False)
        print(logs)
        wandb.log({"eval": logs})