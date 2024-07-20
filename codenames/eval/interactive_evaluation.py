import argparse
from codenames.board import Board
from codenames.embeddings.glove_embeddings import GloveEmbeddings, TrainGuesserEmbeddings
from codenames.game import Game
from codenames.players import RandomGiver, RandomGuesser
from codenames.players.literal_guesser import LiteralGuesser
from codenames.players.match_giver import MatchGiver
from codenames.players.pragmatic_giver import PragmaticGiver
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from codenames.utils.board_utils import make_board_set

plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 28
})
sns.set_style("white", {
    "font.family": "serif",
    "font.weight": "heavy",
    "font.serif": ["Times", "Palatino", "serif"],
    'axes.facecolor': 'white',
    'lines.markeredgewidth': 1})

def play_game_eval(guesser, 
                   giver,
                   num_games,
                   seed=0, 
                   verbose=False):
    """
    Play a game of Codenames with the given guesser and giver.
    
    Args:
        guesser: The guesser to play the game.
        giver: The giver to play the game.
    """
    num_won, num_targets, num_turns = 0, 0, 0
    num_goal_guessed, num_neutral_guesed, num_avoid_guessed = 0, 0, 0
    boards = make_board_set(num_games, seed=1)
    for i in range(num_games):
        board = Board(seed=seed + i)
        board.set_words(*boards[i])
        # print(board.goal_words)
        game = Game(board, giver, guesser, verbose=verbose)
        while game.status == "running":
            clue, targets = game.play_giver()
            results = game.play_guesser(clue, targets)
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
    return {
        "win_rate": num_won / num_games,
        "average_turns_per_game": num_turns / num_games,
        "average_targets_per_turn": num_targets / num_turns,
        "goal_guessed": num_goal_guessed / num_targets,
        "neutral_guessed": num_neutral_guesed / num_targets,
        "avoid_guessed": num_avoid_guessed / num_targets
    }

def make_table_models(models, model_names):
    data = []
    for model, model_name in zip(models, model_names):
        for other_model, other_model_name in zip(models, model_names):
            if model == other_model:
                continue
            guesser_embeddings = TrainGuesserEmbeddings(
                in_embed_dim=300,
                out_embed_dim=1000,
            )
            guesser_embeddings.load_weights(model)
            giver_embeddings = TrainGuesserEmbeddings(
                in_embed_dim=300,
                out_embed_dim=1000,
            )
            giver_embeddings.load_weights(other_model)
            guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)
            giver = PragmaticGiver(giver_embeddings, choose_argmax=True, adaptive=False)
            logs = play_game_eval(guesser, giver, num_games=1, verbose=False)
            data.append([model_name, other_model_name, logs["win_rate"], logs["average_turns_per_game"]])
    df = pd.DataFrame(data, columns=["guesser", "giver", "win_rate", "average_turns_per_game"])

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 28
    })
    sns.set_style("white", {
        "font.family": "serif",
        "font.weight": "heavy",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})
    plt.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
    plt.savefig('../../figures/education.pdf', bbox_inches='tight')
    return df

if __name__ == "__main__":
    # guesser = RandomGuesser()
    # giver = RandomGiver()
    # print("="*100, "\n")
    # print("Random Guesser vs Random Giver")
    # print(play_game_eval(guesser, giver, num_games=10))
    parser = argparse.ArgumentParser()
    parser.add_argument("--giver_embeddings_path", type=str, default="", help="path to trained embeddings for clue giver")
    parser.add_argument("--guesser_embeddings_path", type=str, default="", help="path to trained embeddings for guesser")  
    args = parser.parse_args()

    # guesser_embeddings = TrainEmbeddings(
    #     in_embed_dim=300,
    #     out_embed_dim=1000,
    # )
    # # guesser_embeddings.load_weights("models/education_graduate_300_1000.pth")

    # giver_embeddings = TrainEmbeddings(
    #     in_embed_dim=300,
    #     out_embed_dim=1000,
    # )
    

    embeddings = GloveEmbeddings(embed_dim=300)
    
    models = ["models/education_high_school_associate_300_1000.pth", 
              "models/education_graduate_300_1000.pth", 
              "models/education_bachelor_300_1000.pth",]
    model_names = ["High School/Associate", "Graduate", "Bachelor"]
    

    guesser_embeddings = TrainGuesserEmbeddings(
        in_embed_dim=300,
        out_embed_dim=1000,
    )
    guesser_embeddings.load_weights(models[0])

    giver_embeddings = TrainGuesserEmbeddings(
        in_embed_dim=300,
        out_embed_dim=1000,
    )
    giver_embeddings.load_weights(models[2])

    guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)

    giver = PragmaticGiver(giver_embeddings, choose_argmax=True, adaptive=False)

    test_embeddings = []
    for model in models:
        embeddings = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        embeddings.load_weights(model)
        test_embeddings.append(embeddings)

    match_giver = MatchGiver(test_embeddings[2],
                    [test_embeddings[2]], 
                    choose_argmax=True, 
                    alpha=0.5)
    for embedding, model_name in zip(test_embeddings, model_names):
        if model_name == "High School/Associate":
            continue
        match_giver = MatchGiver(embedding,
                    [embedding], 
                    choose_argmax=True, 
                    alpha=0.5, 
                    max_num_targets=1)
        print("="*100, "\n")
        print(f"High School guesser \n  vs {model_name} giver")
        print(play_game_eval(guesser, match_giver, num_games=1000, verbose=False))
    
    print("="*100, "\n")
    print(f"High School guesser \n  vs Graduate, Bachelor giver")
    match_giver = MatchGiver(test_embeddings[2],
                    test_embeddings[1:], 
                    choose_argmax=True, 
                    alpha=0.3, 
                    max_num_targets=1)
    
    print(play_game_eval(guesser, match_giver, num_games=1000, verbose=False))
    print(match_giver.num_times_chosen)
    # print(play_game_eval(guesser, giver, num_games=25, verbose=False))
    
    # df = make_table_models(models, model_names)
    # for model in models:
    #     for other_model in models:
    #         if model == other_model:
    #             continue
    #         print("="*100, "\n")
    #         print(f"{model} guesser \n  vs {other_model} giver")
    #         guesser_embeddings.load_weights(model)
    #         giver_embeddings.load_weights(other_model)
    #         print(play_game_eval(guesser, giver, num_games=100, verbose=False))

    # guesser = LiteralGuesser(embeddings, choose_argmax=False)

    # giver = PragmaticGiver(guesser, embeddings, choose_argmax=True)
    # print("="*100, "\n")
    # print("No Argmax Literal Guesser vs Argmax Pragmatic Giver")
    # print(play_game_eval(guesser, giver, num_games=10, verbose=False))

    # guesser = LiteralGuesser(embeddings, choose_argmax=True)

    # giver = PragmaticGiver(guesser, embeddings, choose_argmax=False)
    # print("="*100, "\n")
    # print("Argmax Literal Guesser vs Argmax Pragmatic Giver")
    # print(play_game_eval(guesser, giver, num_games=10, verbose=False))

