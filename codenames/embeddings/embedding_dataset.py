from typing import Sequence

from torch.utils.data import Dataset

from scripts.parse_cultural_codes import parse_correct_clues, parse_generate_guess, parse_sub_levels

class GuesserEmbeddingDataset(Dataset):
    def __init__(
        self, 
        clues: Sequence[str],
        guesses: Sequence[Sequence[str]],
        not_guesses: Sequence[Sequence[str]],
        join_words: bool = False,
    ):
        """
        clues: list of strings, the clues that were given
        guesses: list of list of strings, the words that were guessed
        not_guesses: list of list of strings, the words that were not guessed
        """
        assert len(clues) == len(guesses) and len(guesses) == len(not_guesses)
        self.clues = clues
        self.guesses = guesses
        self.not_guesses = not_guesses 
        self.join_words = join_words
    
    def __len__(self):
        return len(self.clues)
    
    def __getitem__(self, idx):
        if self.join_words:
            return self.clues[idx], ",".join(self.guesses[idx]), ",".join(self.not_guesses[idx])
        else:
            return self.clues[idx], self.guesses[idx], self.not_guesses[idx]
        

class GiverEmbeddingDataset(Dataset):
    def __init__(
        self, 
        clues: Sequence[str],
        targets: Sequence[Sequence[str]],
        neutral: Sequence[Sequence[str]],
        avoid: Sequence[Sequence[str]],
        join_words: bool = False,
    ):
        """
        clues: list of strings, the clues that were given
        guesses: list of list of strings, the words that were guessed
        not_guesses: list of list of strings, the words that were not guessed
        """
        assert len(clues) == len(targets) and len(targets) == len(neutral) and len(neutral) == len(avoid)
        self.clues = clues
        self.targets = targets
        self.neutral = neutral
        self.avoid = avoid
        self.join_words = join_words
    
    def __len__(self):
        return len(self.clues)
    
    def __getitem__(self, idx):
        if self.join_words:
            return self.clues[idx], ",".join(self.targets[idx]), ",".join(self.neutral[idx]),  ",".join(self.avoid[idx])
        else:
            return self.clues[idx], self.targets[idx], self.neutral[idx], self.avoid[idx]
        

def get_generate_guess_dataset(split: str) -> GuesserEmbeddingDataset:
    """
    Returns `GuesserEmbeddingDataset` consisting of data from generate guess task.

    `split` can take on train, test, val, or all based on what section of the data you want to load.
    """
    df = parse_generate_guess(split)
    clues = df['clue'].tolist()

    # convert strings and concatenated strings to sequences of strings
    guesses = [[i] for i in df['guess'].tolist()]
    not_guesses = [i.split(', ') for i in df['not_guess'].tolist()]

    dataset = GuesserEmbeddingDataset(clues, guesses, not_guesses)
    return dataset

def get_correct_clues_dataset(split: str) -> GiverEmbeddingDataset:
    """
    Returns `GiverEmbeddingDataset` consisting of data from generate guess task.

    `split` can take on train, test, val, or all based on what section of the data you want to load.
    """
    df = parse_correct_clues(split)
    clues = df['clue'].tolist()

    # convert strings and concatenated strings to sequences of strings
    targets = [i.replace("'", '').split(', ') for i in df['targets'].tolist()]
    neutral = [i.replace("'", '').split(', ') for i in df['tan'].tolist()]
    avoid = [i.replace("'", '').split(', ') for i in df['black'].tolist()]

    dataset = GiverEmbeddingDataset(clues, targets, neutral, avoid)
    return dataset

def make_filter(df, group):
    if group == "AGE under 30":
        return df['age'].isin(('18', '22'))
    elif group == "AGE over 30":
        return df['age'].isin(('30', '45'))
        
    elif group == "NATIVE true":
        return df['native'] == 'true'
    elif group == "NATIVE false":
        return df['native'] == 'false'
        
    elif group == "COUNTRY united states":
        return df['country'] == 'united states'
    elif group == "COUNTRY foreign":
        return df['country'] != 'united states'
    
    elif group == "GENDER male":
        return df['gender'] == 'male'
    elif group == "GENDER female":
        return df['gender'] != 'male'
    
    elif group == "EDUCATION high school associate":
        return df['education'].isin(('high school', 'associate'))
    elif group == "EDUCATION bachelor":
        return df['education'] == 'bachelor'
    elif group == "EDUCATION graduate":
        return df['education'].isin(('master', 'doctorate degree'))
    
    elif group == "EDUCATION not high school associate":
        return df['education'].isin(('bachelor', 'master', 'doctorate degree'))
    elif group == "EDUCATION not bachelor":
        return df['education'].isin(('high school', 'associate', 'master', 'doctorate degree'))
    elif group == "EDUCATION not graduate":
        return df['education'].isin(('high school', 'associate', 'bachelor'))
    
    elif group == "RELIGION catholic":
        return df['religion'] == 'catholicism'
    elif group == "RELIGION not catholic":
        return df['religion'] != 'catholicism'
    
    elif group == "POLITICAL liberal":
        return df['political'].isin(('liberal', 'moderate liberal'))
    elif group == "POLITICAL conservative":
        return df['political'].isin(('moderate conservative', 'conservative', 'libertarian'))
    
    else:
        raise NotImplementedError

def get_generate_guess_dataset_culture_splits(split: str, group: Sequence[str]) -> GuesserEmbeddingDataset:
    """
    Returns `EmbeddingDataset` consisting of data from generate guess task.

    `split` can take on train, test, val, or all based on what section of the data you want to load.
    """
    df = parse_sub_levels(split)
    for g in group:
        df = df[make_filter(df, g)]
    df.dropna(how="any", inplace=True)
    clues = df['clue'].tolist()

    # convert strings and concatenated strings to sequences of strings
    guesses = [[i] for i in df['guess'].tolist()]
    not_guesses = [i.split(', ') for i in df['not_guess'].tolist()]

    dataset = GuesserEmbeddingDataset(clues, guesses, not_guesses)
    return dataset

