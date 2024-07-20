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
from codenames.players.match_giver import MatchGiver
from codenames.players.selective_pragmatic_giver import SelectivePragmaticGiver 
from codenames.eval.interactive_evaluation import play_game_eval
from codenames.utils.board_utils import make_board_set
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
parser.add_argument("--guesser_embeddings_path", type=str, 
                    default="models/education_high_school_associate_300_1000.pth", 
                    help="path to trained embeddings for guesser")
parser.add_argument("--llama-guesser", action=argparse.BooleanOptionalAction, help="use llama guesser")
parser.add_argument("--num_games", type=int, default=101, help="number of games to play")
parser.add_argument("--seed", type=int, default=100, help="random seed")
parser.add_argument("--log_every", type=int, default=25, help="log every n games")
parser.add_argument("--eval_every", type=int, default=10,  help="eval every n games")
args = parser.parse_args()

print(args)

education_models = [
    # "models/education_bachelor_300_1000.pth",
    # "models/education_graduate_300_1000.pth",
    "models/country_foreign_300_1000.pth",
    # "models/country_united_states_300_1000.pth",
    # "models/education_high_school_associate_300_1000.pth",
]

models = [
    "models/education_high_school_associate_300_1000.pth",
    "models/education_graduate_300_1000.pth",
    "models/education_bachelor_300_1000.pth",
    "models/country_foreign_300_1000.pth",
    "models/country_united_states_300_1000.pth",
]

def run_match_experiments(test_models):
    test_embeddings = []
    for model in test_models:
        embeddings = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        embeddings.load_weights(model)
        test_embeddings.append(embeddings)

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

    giver = MatchGiver(test_embeddings[0],
                    test_embeddings, 
                    choose_argmax=True, 
                    alpha=0.3)

    board = Board(seed=args.seed)
    game = TrainingGame(board, 
                giver, 
                guesser, 
                verbose=args.verbose
            )

    num_won, num_targets, num_turns = 0, 0, 0
    num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0

    total_num_won = 0
    boards = make_board_set(args.num_games, seed=args.seed)
    print(boards)
    for i in range(args.num_games):
        board = Board(seed=args.seed + i)
        board.set_words(*boards[i])
        game = TrainingGame(board, 
                    giver, 
                    guesser, 
                    verbose=args.verbose)
        
        while game.status == "running":
            clue, targets = game.play_giver()
            results, giver_logs, guesser_logs = game.play_guesser(clue, targets)
            # print(giver_logs)
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
            total_num_won += 1
        if (i+1) % args.log_every == 0:
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
            print(giver_logs)
            # wandb.log({"train": train_logs})
            num_won, num_targets, num_turns = 0, 0, 0
            num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0
    print("*"*250)
    win_rate = total_num_won / args.num_games
    print("total win rate", win_rate)
    return win_rate
        
    # if i % args.eval_every == 0:
    #     print("="*100, "\n")
    #     print("eval")
    #     with torch.no_grad():
    #         logs = play_game_eval(guesser, eval_giver, num_games=100, seed=7, verbose=False)
    #     print(logs)
    #     wandb.log({"eval": logs})

if __name__ == "__main__":
    models = [
        # "models/education_high_school_associate_300_1000.pth",
        # "models/education_graduate_300_1000.pth",
        # "models/education_bachelor_300_1000.pth",
        "models/country_foreign_300_1000.pth",
        # "models/country_united_states_300_1000.pth",
    ]
    win_rate = run_match_experiments(models)
    # run_match_experiments(models)