import pandas as pd
from IPython import embed
import re

def parse_generate_guess(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the clues, guesses, and not guesses from the generate guesses data.

    Reads in data from train, test, val, or all of them combined based on `split`. Defaults to not
    saving the generated df, but if `save_file` is true, will save to the `data/` directory.
    """
    assert split in ['all', 'train', 'test', 'val']

    # read in raw csv
    source_df = pd.read_csv(f'cultural-codes/codenames/data/generate_guess_task/{split}.csv')

    parse_not_guesses = lambda x: x.split(', hint: ')[0].split('remaining: ')[1].replace('[', '').replace(']', '').replace("'", "").strip()
    parse_clue = lambda x: x.split('hint: ')[1]

    df = pd.DataFrame()
    df['clue'] = source_df['base_text'].apply(parse_clue)
    df['guess'] = source_df['output']
    df['not_guess'] = source_df['base_text'].apply(parse_not_guesses)

    # filter rows where guesses are NaN
    df = df[df["guess"].notna()]
    
    # filter guesses out of not guesses
    for i, row in df.iterrows():
        guesses = row["guess"].split(", ")
        not_guesses = [x for x in row["not_guess"].split(", ") if x not in guesses]
        row["not_guess"] = ", ".join(not_guesses)
    
    # load in GUESSER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = source_df[key].str.split('GUESSER: ').str[1].str.split(']').str[0][0].split('[')[1]
        
    if save_file:
        # save file to data/ directory
        df.to_csv(f'data/generate_guess_{split}.csv', index=False)
    
    return df

def parse_guess_rationale(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Using the clue, guess, and rationale, predict the culture of the GUESSER.
    
    Reads in data from train, test, val, or all of them combined based on `split`. Defaults to not
    saving the generated df, but if `save_file` is true, will save to the `data/` directory.
    """
    assert split in ['all', 'train', 'test', 'val']

    # read in the guess rationale task csv file
    source_df = pd.read_csv(f'cultural-codes/codenames/data/guess_rationale_task/{split}.csv')

    parse_clue = lambda x: x.split('clue: ')[1].split(',')[0].strip()
    parse_guess = lambda x: x.split('guess: ')[1].strip()
    parse_demo = lambda x: x.split('GUESSER: [')[1].split(']')[0]

    df = pd.DataFrame()
    # load in clue, guess, and rationale
    df['clue'] = source_df['base_text'].apply(parse_clue)
    df['guess'] = source_df['base_text'].apply(parse_guess)
    df['rationale'] = source_df['output']
    
    # load in GUESSER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = source_df[key].apply(parse_demo)
    
    if save_file:
        df.to_csv(f'data/guess_rationale_{split}.csv', index=False)

    return df

def parse_correct_guess(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the clues, guesses, and not guesses from the generate guesses data.

    Reads in data from train, test, val, or all of them combined based on `split`. Defaults to not
    saving the generated df, but if `save_file` is true, will save to the `data/` directory.
    """
    assert split in ['all', 'train', 'test', 'val']

    # read in raw csv
    source_df = pd.read_csv(f'cultural-codes/codenames/data/generate_guess_task/{split}.csv')
    target_df = pd.read_csv(f'cultural-codes/codenames/data/correct_guess_task/{split}.csv')

    parse_not_guesses = lambda x: x.split(', hint: ')[0].split('remaining: ')[1].replace('[', '').replace(']', '').replace("'", "").strip()
    parse_clue = lambda x: x.split('hint: ')[1]
    parse_target = lambda x: x.split('target: ')[1].split(', hint:')[0].strip()

    df = pd.DataFrame()
    df['clue'] = source_df['base_text'].apply(parse_clue)
    df['guess'] = source_df['output']
    df['not_guess'] = source_df['base_text'].apply(parse_not_guesses)
    df['target'] = target_df['base_text'].apply(parse_target)
    
    # load in GUESSER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = source_df[key].str.split('GUESSER: ').str[1].str.split(']').str[0][0].split('[')[1]
        
    if save_file:
        # save file to data/ directory
        df.to_csv(f'data/correct_guess_{split}.csv', index=False)
    
    return df

def parse_select_target(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the goal words, avoid words, neutral words and targets from generate target data.
    
    Reads in data from train, test, val, or all of them combined based on `split`. Defaults to not
    saving the generated df, but if `save_file` is true, will save to the `data/` directory.

    """

    # read in raw csv
    # source_df = pd.read_csv(f'cultural-codes/codenames/data/generate_guess_task/{split}.csv')
    target_df = pd.read_csv(f'cultural-codes/codenames/data/correct_guess_task/{split}.csv')

    parse_remaining = lambda x: x.split('[')[1].split(']')[0].replace("'", '').strip()
    parse_clue = lambda x: x.split('hint: ')[1]
    parse_target = lambda x: x.split('target: ')[1].split(', hint:')[0].strip()

    df = pd.DataFrame()
    df['clue'] = target_df['base_text'].apply(parse_clue)
    df['remaining'] = target_df['base_text'].apply(parse_remaining)
    df['target'] = target_df['base_text'].apply(parse_target)
    
    # load in GUESSER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = target_df[key].apply(lambda x: x.split('GUESSER: [')[1].split(']')[0])
        
    if save_file:
        # save file to data/ directory
        df.to_csv(f'data/correct_guess_{split}.csv', index=False)
    
    return df

def split_string_into_dict(str):
    pairs = str.split('],')
    dct = {}
    for pair in pairs:
        spl = pair.split(':')
        value = [item.strip("' \n[]") for item in spl[1].strip('[]').split(', ')]
        dct[spl[0].strip(" \n")] = value
    return dct

def parse_string(input_string):
    # Regular expression pattern to extract key-value pairs
    pattern = r"\b(\w+):\s*([0-9.]+)|\b(\w+):\s*(\w+\s*\w*)"

    # Find all matches in the input string
    matches = re.findall(pattern, input_string)
    # Construct dictionary from matches
    result = {}
    for match in matches:
        if match[0]:
            key = match[0]
            value = match[1]
        else:
            key = match[2]
            value = match[3]
        result[key] = value
    return result

def parse_correct_targets(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the goal words, avoid words, neutral words and targets from generate target data.
    """
    target_df = pd.read_csv(f'cultural-codes/codenames/data/target_selection_task/{split}.csv')
    
    df = pd.DataFrame()
    df['targets'] = target_df['output']
    
    for i, row in target_df.iterrows():
        total = []
        for key, value in split_string_into_dict(row['base_text']).items():
            total += value
            df.loc[i, key] = str(value).replace("'", '').strip('[]')
        df.loc[i, 'total'] = str(total)

     # load in GIVER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = target_df[key].str.split('GIVER: ').str[1].str.split(']').str[0][0].split('[')[1]

    for i, row in target_df.iterrows():
        for culture_type in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
            for key, value in parse_string(row[culture_type]).items():
                df.loc[i, key] = value
    return df

def parse_correct_clues(split:str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the clue, target, avoid, and neutral words from generate clue data.
    """
    clue_df = pd.read_csv(f'cultural-codes/codenames/data/clue_generation_task/{split}.csv')

    # parse_target = lambda x: x.split('target: ')[1].split(', hint:')[0].strip()

    df = pd.DataFrame()
    df['clue'] = clue_df['output']
    # df['target'] = clue_df['base_text'].apply(parse_target)


    for i, row in clue_df.iterrows():
        total = []
        for key, value in split_string_into_dict(row['base_text']).items():
            total += value
            df.loc[i, key] = str(value).replace("'", '').strip('[]')
        df.loc[i, 'total'] = str(total)
    
     # load in GIVER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = clue_df[key].str.split('GIVER: ').str[1].str.split(']').str[0][0].split('[')[1]

    for i, row in clue_df.iterrows():
        for culture_type in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
            for key, value in parse_string(row[culture_type]).items():
                df.loc[i, key] = value
    return df
    
    
    

def parse_sub_levels(split: str, save_file: bool = False) -> pd.DataFrame:
    """
    Parse the clues, guesses, and not guesses from the generate guesses data.
    Same as parse_generate_guess but for sub-levels of demographics such as gender, country, etc.
    """
    assert split in ['all', 'train', 'test', 'val']
    # read in raw csv
    source_df = pd.read_csv(f'cultural-codes/codenames/data/generate_guess_task/{split}.csv')
    target_df = pd.read_csv(f'cultural-codes/codenames/data/correct_guess_task/{split}.csv')

    parse_not_guesses = lambda x: x.split(', hint: ')[0].split('remaining: ')[1].replace('[', '').replace(']', '').replace("'", "").strip()
    parse_clue = lambda x: x.split('hint: ')[1]
    parse_target = lambda x: x.split('target: ')[1].split(', hint:')[0].strip()

    df = pd.DataFrame()
    df['clue'] = source_df['base_text'].apply(parse_clue)
    df['guess'] = source_df['output']
    df['not_guess'] = source_df['base_text'].apply(parse_not_guesses)
    df['target'] = target_df['base_text'].apply(parse_target)

    # filter rows where clues or guesses are NaN
    df.dropna(how="any", inplace=True)
    
    # load in GUESSER demographics
    for key in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
        df[key] = source_df[key].str.split('GUESSER: ').str[1].str.split(']').str[0][0].split('[')[1]

    for i, row in source_df.iterrows():
        for culture_type in ['leaning_only', 'event_only', 'demo_only', 'personality_only']:
            for key, value in parse_string(row[culture_type]).items():
                df.loc[i, key] = value
    return df
