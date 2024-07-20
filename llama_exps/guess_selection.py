import fire
import time
import argparse
import pandas as pd
import numpy as np
import ast

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_exps.utils import add_player_demo_col, get_fixed_generation_prob, load_model_and_tokenizer, ALL_DEMOS, HF_BATCH_SIZE

def read_in_prompts(split: str, demo: str, task: str):
    # the `generate_guess` task means we calculate accuracy based on the guesser's output
    # the `correct_guess` task means we calculate accuracy based on the giver's intended output
    assert task in ['generate_guess', 'correct_guess']
    df = pd.read_csv(f'cultural-codes/codenames/data/{task}_task/{split}.csv')
    df = add_player_demo_col(demo=demo, df=df, is_giver=True) # we want to add demographic info about the giver

    data = df['base_text'].tolist()
    indices = df[df.columns[0]].tolist() # indices for cross validation
    demo_texts = df['demo_text'].tolist()

    # read "targets" based on task type (guesser vs. giver alignment)
    if task == 'generate_guess':
        targets = df['output'].tolist()
    elif task == 'correct_guess':
        targets = df['base_text'].apply(lambda x: x.split('target: ')[1].split('hint:')[0]).tolist()

    words, hints = [], []
    for d in data:
        opts, hint = d.split('hint:')
        hint = hint.strip()

        opts = opts.split('[')[1].split(']')[0].replace("'", "")
        opts_lst = opts.split(',')
        opts_lst = [c.strip() for c in opts_lst]

        hints.append(hint)
        words.append(opts_lst)

    assert len(hints) == len(words) == len(demo_texts)

    prompts = []
    for w, h, d_text in zip(words, hints, demo_texts):
        formatted_words = ', '.join(w[:-1]) + ', and ' + w[-1]
        prompt_text = f"You are playing Codenames. The possible words are {formatted_words}."
        extractor_text = f"For the hint {h}, the most likely target word is "

        to_join = [prompt_text, d_text, extractor_text] if demo else [prompt_text, extractor_text]
        
        prompt = " ".join(to_join)
        prompts.append(prompt)

    assert len(prompts) == len(words) == len(targets)
    return prompts, words, targets, indices

def generate_preds(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, split: str, demo: str, gen_dir: str, task: str):
    prompts, words, targets, indices = read_in_prompts(split=split, demo=demo, task=task)

    repeated_prompts, repeated_responses = [], []
    for p, w in zip(prompts, words):
        repeated_prompts.extend([p] * len(w))
        repeated_responses.extend(w)

    output_lps = get_fixed_generation_prob(model,
                              tokenizer,
                              sequences=repeated_prompts,
                              responses=repeated_responses,
                              batch_size=HF_BATCH_SIZE)
    lps = []
    for w in words:
        num_words = len(w)
        prompt_lps, output_lps = output_lps[:num_words], output_lps[num_words:]
        curr_resps, repeated_responses = repeated_responses[:num_words], repeated_responses[num_words:]

        # store probabilities of each target word for future reranking
        zipped = [item for item in zip(curr_resps, prompt_lps)]
        lps.append(zipped)

    assert len(lps) == len(prompts)
    assert len(repeated_responses) == 0
    assert len(output_lps) == 0

    results_df = pd.DataFrame({
        'indices': indices,
        'prompt': prompts,
        'words': words,
        'lps': lps,
        'targets': targets
    })
    
    # save df for debugging
    shorter_task = task.split('_')[0]
    filename = f'guess_target_preds_{shorter_task}_{split}_{demo}'
    results_df.to_csv(f'{gen_dir}/{filename}.csv', index=False)

    return results_df

def main(model_name: str,
         task: str,
         gen_dir: str = 'results', 
         generate: bool = False, 
         evaluate: bool = False,
         split: str = 'val'):
    assert split in ['all', 'test', 'train', 'val']

    start = time.time()

    dfs = {}
    for key in ALL_DEMOS:
        dfs[key] = None

    if generate:
        model, tokenizer = load_model_and_tokenizer(model_name)

        # generate predictions        
        for demo in ['all_text']:
            start_time = time.time()
            print(f"Starting {demo}")
            dfs[demo] = generate_preds(model=model, tokenizer=tokenizer, split=split, demo=demo, gen_dir=gen_dir, task=task)
            end_time = time.time()
            print(f"Finished generating {demo} in {(end_time - start_time) / 60} minutes.")
    else:
        # read in predictions
        for demo in ALL_DEMOS:
            shorter_task = task.split('_')[0]
            filename = f'guess_target_preds_{shorter_task}_{split}_{demo}'
            dfs[demo] = pd.read_csv(f'{gen_dir}/{filename}.csv')

    if evaluate:
        # note that we're computing alignment with human guesser, not whether the guess was
        # actually what the giver intended
        for key, df in dfs.items():
            preds = [ast.literal_eval(i) for i in df['lps'].tolist()]
            targets = df['targets'].tolist()
            
            targets = [[] if (isinstance(t, float) and np.isnan(t)) else t.split(',') for t in targets]
            targets = [[t.strip() for t in ts] for ts in targets]

            # check pred in target bc target is a list of possible clues
            correct, total = 0, 0
            for pred, targ in zip(preds, targets):
                # there are some cases where the target is unspecified in the original data
                if isinstance(targ, float) and pd.isna(targ):
                    print("skipping something")
                    continue
                
                # sort pred based on lps
                pred.sort(key = lambda x: -x[1])
                correct_preds = [p[0] in targ for p in pred[:len(targ)]]

                correct += sum(correct_preds)
                total += len(correct_preds) # note that this supports having multiple target words
            acc = 100 * (correct / total)
            print(f"Accuracy for {key}: {acc:.2f}")
    
    print(f"Finished in {(time.time() - start) / 60} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--gen_dir', nargs='?', type=str, help='Directory to save predictions to', default='results')
    parser.add_argument('--generate', action='store_true', help='Whether to generate predictions')
    parser.add_argument('--evaluate', action='store_true', help='Whether to evaluate predictions')
    parser.add_argument('--split', nargs='?', type=str, help='Subset of data to consider', default='val')
    parser.add_argument('--task', type=str, help='Task type is either correct_guess or generate_guess')
    args = parser.parse_args()

    fire.Fire(main)