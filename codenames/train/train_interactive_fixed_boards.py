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
parser.add_argument("--giver_embeddings_path", type=str, default="models/education_graduate_300_1000.pth", help="path to trained embeddings for clue giver")
parser.add_argument("--guesser_embeddings_path", type=str, default="models/education_high_school_associate_300_1000.pth", help="path to trained embeddings for guesser")
parser.add_argument("--llama-guesser", action=argparse.BooleanOptionalAction, help="use llama guesser")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--log_every", type=int, default=5, help="log every n games")
parser.add_argument("--eval_every", type=int, default=1,  help="eval every n games")
parser.add_argument("--num_games_per_round", type=int, default=10, help="number of games to play per round")
parser.add_argument("--num_rounds", type=int, default=1000, help="number of rounds")
parser.add_argument("--wandb", action="store_true", help="log to wandb")
args = parser.parse_args()
print(args)

if args.wandb:
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

if args.llama_guesser:
    guesser = LlamaGuesser()
else: 
    guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)

giver = PragmaticGiver(giver_embeddings,
                        choose_argmax=False, 
                        k=5, 
                        adaptive=True, 
                        batch_size=args.batch_size,)


board = Board(seed=args.seed)
game = TrainingGame(board, 
            giver, 
            guesser, 
            verbose=args.verbose
        )

def make_board_set(num_boards, seed):
    boards = []
    for i in range(num_boards):
        board = Board(seed=seed + i)
        boards.append([board.goal_words, board.avoid_words, board.neutral_words])
    return boards

boards = make_board_set(args.num_games_per_round, args.seed)
def training_round(giver, guesser, args, boards):
    num_games = 0
    num_won, num_targets, num_turns = 0, 0, 0
    num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0
    for i in range(args.num_games_per_round):
        board = Board(seed=args.seed + i)
        board.set_words(*boards[i])
        game = TrainingGame(board, 
                    giver, 
                    guesser, 
                    verbose=args.verbose)
        
        while game.status == "running":
            clue, targets = game.play_giver()
            results, giver_logs, guesser_logs = game.play_guesser(clue, targets)
            if giver_logs["train_loss"]:
                print(giver_logs)
                if args.wandb:
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
        num_games += 1
    logs = {
                "iter": i,
                "win_rate": num_won / num_games,
                "average_turns_per_game": num_turns / num_games,
                "average_targets_per_turn": num_targets / num_turns,
                "goal_guessed": num_goal_guessed / num_targets,
                "neutral_guessed": num_neutral_guesed / num_targets,
                "avoid_guessed": num_avoid_guessed / num_targets
            }
    return logs, giver

def eval_round(giver, guesser, args, boards):
    eval_giver = PragmaticGiver(guesser, giver.embeddings, choose_argmax=False, k=5, adaptive=False)
    assert giver.embeddings is eval_giver.embeddings
    with torch.no_grad():
        logs = training_round(eval_giver, guesser, args, boards)
    return logs
    

for i in range(args.num_rounds):
    print("training!")
    train_logs, giver = training_round(giver, guesser, args, boards)
    if args.wandb:
        wandb.log({"train": train_logs, "round": i})
    print(train_logs)
    if i % args.eval_every == 0:
        print("="*100, "\n")
        print("eval")
        eval_logs, eval_giver = eval_round(giver, guesser, args, boards)
        print(eval_logs)
        if args.wandb:
            wandb.log({"eval": eval_logs, "round": i})