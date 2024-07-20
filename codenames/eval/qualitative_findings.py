import argparse
import pandas as pd
import time
import itertools
from codenames.embeddings.glove_embeddings import GloveEmbeddings, TrainGuesserEmbeddings
from codenames.players.literal_guesser import LiteralGuesser
from scripts.parse_cultural_codes import parse_generate_guess, parse_select_target, parse_correct_targets, parse_correct_clues
import torch


def check_accuracy_save_differences(embeddings1, embeddings2, clue_data, target_data):

    guesser1 = LiteralGuesser(embeddings1)
    guesser2 = LiteralGuesser(embeddings2)

    differences_df = pd.DataFrame(columns=["score", "clue", "targets", "guesses1", "guesses2", "possibilities"])

    for i, row in target_data.iterrows():
        targets = row["targets"].split(", ")
        avoid = row["black"].split(", ")
        goal = row["green"].split(", ")
        neutral = row["tan"].split(", ")
        lst = avoid + goal + neutral
        clue_row = clue_data.iloc[i]
        assert clue_row["tan"] == row["tan"] and clue_row["black"] == row["black"] and clue_row["targets"] == row["targets"]
        guess1 = guesser1.make_guess(lst, clue_row['clue'], choose_argmax=True, num_targets=len(targets))
        guess2 = guesser2.make_guess(lst, clue_row['clue'], choose_argmax=True, num_targets=len(targets))
        if set(guess1).intersection(set(guess2)) == set() \
            and (set(guess1) == set(targets) and set(guess2) != set(targets))\
                 and set(guess2).intersection(set(goal)) == set():
            similarity1 = guesser1.guess_clue_similarities([clue_row['clue']], guess1)
            similarity2 = guesser2.guess_clue_similarities([clue_row['clue']], guess2)
            score = abs(similarity1 - similarity2)
            differences_df = differences_df._append({
                "score": score,
                "clue": clue_row['clue'],
                "targets": row['targets'],
                "guesses1": guess1,
                "guesses2": guess2, 
                "possibilities": lst,
            }, ignore_index=True)
    differences_df = differences_df.sort_values(by="score", ascending=False)
    differences_df = differences_df.reset_index(drop=True)
    return differences_df


def three_way_differences(embeddings1, embeddings2, embeddings3, clue_data, target_data):
    guesser1 = LiteralGuesser(embeddings1)
    guesser2 = LiteralGuesser(embeddings2)
    guesser3 = LiteralGuesser(embeddings3)

    differences_df = pd.DataFrame(columns=["score", "clue", "targets", "guesses1", "guesses2", "guesses3", "possibilities"])

    for i, row in target_data.iterrows():
        targets = row["targets"].split(", ")
        avoid = row["black"].split(", ")
        goal = row["green"].split(", ")
        neutral = row["tan"].split(", ")
        lst = avoid + goal + neutral
        clue_row = clue_data.iloc[i]
        assert clue_row["tan"] == row["tan"] and clue_row["black"] == row["black"] and clue_row["targets"] == row["targets"]
        guess1 = guesser1.make_guess(lst, clue_row['clue'], choose_argmax=True, num_targets=len(targets))
        guess2 = guesser2.make_guess(lst, clue_row['clue'], choose_argmax=True, num_targets=len(targets))
        guess3 = 
        if set(guess1).intersection(set(guess2)) == set() \
            and set(guess1).intersection(set(guess3)) == set() \
            and (set(guess1) == set(targets) and set(guess2) != set(targets))\
                 and set(guess2).intersection(set(goal)) == set() \
                    and set(guess3).intersection(set(goal)) == set():
            similarity1 = guesser1.guess_clue_similarities([clue_row['clue']], guess1)
            similarity2 = guesser2.guess_clue_similarities([clue_row['clue']], guess2)
            similarity3 = guesser3.guess_clue_similarities([clue_row['clue']], guess3)
            score = (abs(similarity1 - similarity2) + abs(similarity1 - similarity3))/2
            differences_df = differences_df._append({
                "score": score,
                "clue": clue_row['clue'],
                "targets": row['targets'],
                "guesses1": guess1,
                "guesses2": guess2, 
                "possibilities": lst,
            }, ignore_index=True)
    differences_df = differences_df.sort_values(by="score", ascending=False)
    differences_df = differences_df.reset_index(drop=True)
    return differences_df



if __name__ == "__main__":

    models = ["models/education_high_school_associate_300_1000.pth", 
              "models/education_graduate_300_1000.pth", 
              "models/education_bachelor_300_1000.pth",]
    
    # models = [
    #     "models/age_over_30_300_1000.pth",
    #     "models/age_under_30_300_1000.pth",
    # ]
    
    combinations = itertools.combinations(models, 2)
    for model1, model2 in combinations:
        embeddings1 = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        embeddings1.load_weights(model1)

        embeddings2 = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        embeddings2.load_weights(model2)

        # target_df = parse_select_target("val")
        # clue_df = parse_correct_clues("val")
        clue_df = pd.read_csv("cultural-codes/codenames/data/edited_data/clue_df.csv")
        target_df = pd.read_csv("cultural-codes/codenames/data/edited_data/target_df.csv")
        # target_df = pd.merge(clue_df, parse_correct_targets("val"), on="tan")

        differences_df = check_accuracy_save_differences(embeddings1, embeddings2, clue_df, target_df)

        model1_name = model1.split("/")[-1].split(".")[0][10:-10]
        model2_name = model2.split("/")[-1].split(".")[0][10:-10]

        differences_df.to_csv(f"results/differences{model1_name}_{model2_name}.csv")

        differences_df = check_accuracy_save_differences(embeddings2, embeddings1, clue_df, target_df)
        differences_df.to_csv(f"results/differences{model2_name}_{model1_name}.csv")
    

    embeddings1 = TrainGuesserEmbeddings(
        in_embed_dim=300,
        out_embed_dim=1000,
    )
    embeddings1.load_weights("models/education_high_school_associate_300_1000.pth")
    embeddings2 = TrainGuesserEmbeddings(
        in_embed_dim=300,
        out_embed_dim=1000,
    )
    embeddings2.load_weights("models/education_graduate_300_1000.pth")
    embeddings3 = TrainGuesserEmbeddings(
        in_embed_dim=300,
        out_embed_dim=1000,
    )
    embeddings3.load_weights("models/education_bachelor_300_1000.pth")

    diff3_df = three_way_differences(embeddings1, embeddings2, embeddings3, clue_df, target_df)
    

    # embeddings1 = TrainGuesserEmbeddings()
    # embeddings1.load_weights(args.embeddings1)

    # embeddings2 = TrainGuesserEmbeddings()
    # embeddings2.load_weights(args.embeddings2)

    # guess_df = parse_generate_guess("val")

    # differences_df = check_accuracy_save_differences(embeddings1, embeddings2, guess_df)

    # pd.to_csv("results/differences.csv")