import pandas as pd
from codenames.players.literal_guesser import LiteralGuesser
from codenames.embeddings.glove_embeddings import GloveEmbeddings
from codenames.players.pragmatic_giver import PragmaticGiver
from scripts.parse_cultural_codes import parse_generate_guess, parse_sub_levels, parse_correct_clues, parse_correct_targets, parse_select_target
import numpy as np
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
from IPython import embed
import json
from tqdm import tqdm

def check_agreement_old(guesser, data, num_targets=2):
    """
    Check the agreement between the guesser and the data.
    
    Args:
        guesser: The guesser to check the agreement with.
        data: The data to check the agreement with.
    """
    correct, correct_targets = 0, 0
    total_guesses, total_targets = 0, 0
    human_guess_correct = 0
    for i, row in data.iterrows():
        st = set(row['not_guess'].split(', '))
        human_guesses = set(row['guess'].split(', '))
        targets = row['target'].split(', ')
        st.update(human_guesses)
        st.update(targets)
        lst = list(st)
        for target in targets:
            assert target in lst
        guess = guesser.make_guess(lst, row['clue'], choose_argmax=True, num_targets=num_targets)
        # print(targets, guess)
        for item in human_guesses:
            if item in guess:
                correct += 1/len(human_guesses)
        total_guesses += 1
        for item in targets:
            if item in guess:
                correct_targets += 1/len(targets)
            if item in human_guesses:
                human_guess_correct += 1/len(targets)
        total_targets += 1
    return {
        "correct_accuracy": correct / total_guesses,
        "target_accuracy": correct_targets / total_targets,
        "human_guess_accuracy": human_guess_correct / total_targets
    }

def check_agreement_new(guesser, data):
    """
    Check the agreement between the guesser and the data.
    
    Args:
        guesser: The guesser to check the agreement with.
        data: The data to check the agreement with.
    """

    correct, total_guesses = 0, 0
    for i, row in data.iterrows():
        st = set(row['not_guess'].split(', '))
        human_guesses = set(row['guess'].split(', '))
        num_human_guesses = len(human_guesses)
        st.update(human_guesses)
        possible_words = list(st)
        pred_human_guesses = guesser.make_guess(possible_words, row['clue'], choose_argmax=True, num_targets=num_human_guesses)
        
        for item in human_guesses:
            if item in pred_human_guesses:
                correct += 1
        total_guesses += num_human_guesses
    return {
        "guess_accuracy": correct / total_guesses,
    }

def check_guesser_target_accuracy(guesser, data):
    """
    Check the agreement between the guesser and the targets given by human clue givers.
    
    Args:
        guesser: The guesser to check the agreement with.
        data: The data to check the agreement with.
    """
    correct_targets, total_targets = 0, 0
    with open('assets/clue_list.json') as f:
        possible_clues = json.load(f)
    for i, row in tqdm(data.iterrows()):
        targets = row["target"].split(", ")
        unselected = list(set(row["remaining"].split(", ")) | set(targets))
        # print(targets)
        # print(unselected)
        num_targets = len(targets)
        for target in targets:
            assert target in unselected
        pred_targets = guesser.make_guess(unselected, row['clue'], choose_argmax=True, num_targets=num_targets)
        for target in targets:
            if target in pred_targets:
                correct_targets += 1
        total_targets += num_targets
    return {
        "target_accuracy": correct_targets / total_targets
    }

def check_clue_agreement(giver, data):
    """
    Check the agreement between the giver and the clues selected by human clue givers.
    
    Args:
        guesser: The guesser to check the agreement with.
        data: The data to check the agreement with.
    """
    correct_clues, total_clues = 0, 0
    with open('assets/clue_list.json') as f:
        possible_clues = json.load(f)
    for i, row in tqdm(data.iterrows()):
        targets = row['targets'].split(', ')
        avoid = row['black'].split(', ')
        neutral = row['tan'].split(', ')
        human_clue = row["clue"]
        #TODO: get list of possible clues
        pred_clue = giver.give_clue(targets, avoid, neutral, possible_clues, targets, choose_argmax=True)
        if pred_clue == human_clue:
            correct_clues += 1
        total_clues += 1
    return {
        "clue_accuracy": correct_clues / total_clues,
    }

def check_target_selection_agreement(giver, data):
    """
    Check the agreement between the clue giver selected targets and the targets selected in the dataset.
    """
    correct_targets, total_targets = 0, 0
    with open('assets/clue_list.json') as f:
        possible_clues = json.load(f)
    for i, row in tqdm(data.iterrows()):
        targets = row["targets"].split(", ")
        avoid = row["black"].split(", ")
        goal = row["green"].split(", ")
        neutral = row["tan"].split(", ")
        num_targets = len(targets)
        pred_targets = giver.select_targets(goal, avoid, neutral, possible_clues, num_targets=num_targets)
        for target in targets:
            if target in pred_targets:
                correct_targets += 1
        total_targets += num_targets
    return {
        "target_accuracy": correct_targets / total_targets
    }

def check_based_on_values(data, key, guesser, num_targets=2, percent_threshold=0.05):
    """
    Check the agreement based on the values of a key.
    
    Args:
        data: The data to check the agreement with.
        key: The key to check the agreement based on.
        guesser: The guesser to check the agreement with.
    """
    values = np.unique(data[[key]].values)
    for value in values:
        print(f"{key}: {value}")
        percent_data = len(data[data[key] == value])/len(data)
        if percent_data < percent_threshold:
            continue
        print("percentage of data: ", percent_data)
        results = check_agreement_new(guesser, data[data[key] == value])
        print(results)

if __name__ == "__main__":

    print("new target selection and guess accuracy")

    embeddings = GloveEmbeddings(embed_dim=300)
    guesser = LiteralGuesser(embeddings)

    target_df = parse_select_target("val")
    print("="*100 + "\n")
    print("target accuracy")
    print(check_guesser_target_accuracy(guesser, target_df))

    guess_df = parse_generate_guess("val")
    print("="*100 + "\n")
    print("guess accuracy")
    print(check_agreement_new(guesser, guess_df))



    # print("Compute Human Agreement for Clue and Target Selection")
    # clue_df = parse_correct_clues("val")
    # target_df = parse_correct_targets("val")

    # embeddings = GloveEmbeddings(embed_dim=300)
    # guesser = LiteralGuesser(embeddings)

    # giver = PragmaticGiver(guesser, embeddings)
    # print(check_clue_agreement(giver, clue_df))
    # print(check_target_selection_agreement(giver, target_df))

    # df = parse_sub_levels('val')
    # df.dropna(how="any", inplace=True)
    # print(len(df))

    # correct_targets = df[df["target"] == df["guess"]].shape[0]
    # print(correct_targets/len(df))

    # print("GloVe")
    # embeddings = GloveEmbeddings(embed_dim=300)
    # guesser = LiteralGuesser(embeddings)
    # print("New Agreement", check_agreement_new(guesser, df))

    # keys = set(df.keys()) - set(["clue", "guess", "not_guess", "target", "leaning_only", "event_only", "demo_only", "personality_only", "hint"])
    # keys = ["age"]
    # for key in keys:
    #     print("-"*50, f"Based on {key}", "-"*50)
    #     check_based_on_values(df, key, guesser, percent_threshold=0.1)

    # print("-"*50, "Based on Demogrpahics", "-"*50)
    # print("Political")
    # political_values = np.unique(df[["political"]].values)
    # for value in political_values:
    #     print(f"Political: {value}")
    #     print(check_agreement_new(guesser, df[df["political"] == value]))
    # # load in GUESSER demographics
    # for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
    #     df[key] = source_df[key].str.split('GUESSER: ').str[1].str.split(']').str[0][0].split('[')[1]


