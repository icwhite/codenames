import fire
import time
import argparse
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_exps.utils import add_player_demo_col, ALL_DEMOS, load_model_and_tokenizer, hf_llama_generate

def generate_clues(model: AutoModelForCausalLM, 
                   tokenizer: AutoTokenizer,
                   split: str, 
                   demo: str, 
                   save_df: bool = True, 
                   logprobs: bool = False, 
                   n_samples: int = None,
                   save_dir: str = 'results'):
    prompts = []
    df = pd.read_csv(f'cultural-codes/codenames/data/clue_generation_task/{split}.csv')
    if n_samples:
        df = df[:n_samples]

    # add demographic column for the guesser
    df = add_player_demo_col(demo=demo, df=df, is_giver=False)
        
    data = df['base_text'].tolist()
    indices = df[df.columns[0]].tolist()
    demo_texts = df['demo_text'].tolist()

    assert len(data) == len(demo_texts)
    for d, d_text in zip(data, demo_texts):
        card_types = ['black', 'tan', 'targets']
        splits = []
        for card_type in card_types:
            words = d.split(f"{card_type}: [")[1].split("]")[0].replace("'", "").split(",")
            words = [w.strip() for w in words]
            splits.append(words)

        avoid_words = ", ".join(splits[0])
        neutral_words = ", ".join(splits[1])
        target_words = ", ".join([f"'{i}'" for i in splits[2]])

        prompt_text = f"You are playing Codenames. You can only give clues which are one word. One clue will apply to multiple targets. Words to avoid are {avoid_words}. Neutral words are {neutral_words}."
        few_shot_text = f"For the group of target words ['fall', 'spring', and 'leaf'] the best clue is 'season'. For the group of target words ['round', 'cylinder'] the best clue is 'circle'. For the target words [{target_words}] the best clue is '"

        # only add demographic information if 1) the demographic column isn't None 2) there is demographic information
        # provided for the guesser
        to_join = [prompt_text, d_text, few_shot_text] if demo and d_text != '' else [prompt_text, few_shot_text]
        
        prompt = " ".join(to_join)
        prompts.append(prompt)

    # allow the model to generate a couple of tokens for more complex clues. we'll parse up until the ending quotation mark
    batch_size = 8
    clue_gens, clue_lps = hf_llama_generate(model, tokenizer, prompts, max_new_tokens=3, batch_size=batch_size)
    clue_gens = [i.split("'")[0] if "'" in i else i for i in clue_gens]

    df = pd.DataFrame({
        'indices': indices,
        'prompt': prompts,
        'clues': clue_gens,
    })

    if logprobs:
        df = df.assign(logprobs=clue_lps)

    if save_df:
        df.to_csv(f"{save_dir}/clue_generation_results_{demo}_{n_samples}.csv", index=False)

    return df

def score_clues(pred_df: pd.DataFrame, split: str):
    # Note gold_df and pred_df may not be the same size if we limit to `n_samples`
    gold_df = pd.read_csv(f'cultural-codes/codenames/data/clue_generation_task/{split}.csv')[:len(pred_df)]

    gold = gold_df['output'].tolist()
    pred = pred_df['clues'].tolist()
    pred = [i.split("'")[0] for i in pred]

    correct = [g == p for g, p in zip(gold, pred)]
    return sum(correct) / len(correct)

def main(model_name: str, split: str = 'val', n_samples: int = None):
    assert split in ['all', 'test', 'train', 'val']

    start = time.time()

    model, tokenizer = load_model_and_tokenizer(model_name)

    for demo in ALL_DEMOS:
        gen_df = generate_clues(model=model, tokenizer=tokenizer, demo=demo, split=split, n_samples=n_samples)

        # How does Llama's clues compare to human clues?
        acc = score_clues(gen_df, split)

        print(f"Clue alignment with humans for demo={demo}: {acc * 100 :.2f}%")
        
    print(f"Finished in {(time.time() - start) / 60} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--split', nargs='?', type=str, help='Subset of data to consider', default='val')
    parser.add_argument('--save_dir', nargs='?', type=str, help='Directory to save predictions to', default='results')
    parser.add_argument('--n_samples', nargs='?', type=int, help='Number of samples', default=None)
    args = parser.parse_args()

    fire.Fire(main)