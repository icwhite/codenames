import fire
import time
import argparse
import pandas as pd

from llama_exps.utils import load_model_and_tokenizer, get_fixed_generation_prob

ALL_DEMOS = ['', 'political', 'leaning_only', 'event_only', 'demo_only', 'personality_only', 'all_text']

def read_in_prompts(split: str, n_samples: int = None):
    df = pd.read_csv(f'cultural-codes/codenames/data/target_selection_task/{split}.csv')
    data = df['base_text'].tolist()

    splits = []
    for d in data:
        word_splits = {'green': None, 'black': None, 'tan': None}
        for key in word_splits.keys():
            phrase = d.split(f'{key}: [')[1].split(']')[0].replace("'", '')
            word_splits[key] = phrase
        splits.append(word_splits)

    prompts = []
    for d in splits:
        prompt = f"You are playing Codenames and need to select a target word for your partner to guess. Words to avoid are {d['black']}. Neutral words are {d['tan']}. Goal words are {d['green']}. The best target word for your partner to guess is '"
        
        prompts.append(prompt)

    opts = [[i.strip() for i in s['green'].replace("'", "").split(',')] for s in splits]
    gold_targets = df['output'].tolist()

    if n_samples:
        prompts = prompts[:n_samples]
        opts = opts[:n_samples]
        gold_targets = gold_targets[:n_samples]

    return prompts, opts, gold_targets

def main(model_name: str,
         gen_dir: str = 'results', 
         split: str = 'val',
         n_samples: int = None):
    assert split in ['all', 'test', 'train', 'val']

    start = time.time()

    model, tokenizer = load_model_and_tokenizer(model_name)

    # get prompts
    prompts, opts, gold_targets = read_in_prompts(split=split, n_samples=n_samples)

    repeated_prompts, repeated_responses = [], []
    for p, opt in zip(prompts, opts):
        repeated_prompts.extend([p] * len(opt))
        repeated_responses.extend(opt)

    # get probability of each target word
    output_lps = get_fixed_generation_prob(model,
                           tokenizer,
                           sequences=repeated_prompts, 
                           responses=repeated_responses)

    # get top k=2 targets
    TOP_K = 2
    top_targets = []
    for opt in opts:
        curr_probs, output_lps = output_lps[:len(opt)], output_lps[len(opt):]

        probs_and_opts = list(zip(curr_probs, opt))
        probs_and_opts.sort(key = lambda x: -x[0])

        targets = [p[1] for p in probs_and_opts[:TOP_K]]
        top_targets.append(targets)

    assert len(output_lps) == 0

    df = pd.DataFrame({
        'prompts': prompts,
        'opts': opts,
        'top_targets': top_targets
    })

    df.to_csv(f'{gen_dir}/target_selection.csv', index=False)

    # compute accuracy
    total, correct = 0, 0
    assert len(top_targets) == len(gold_targets)
    for pred, gold in zip(top_targets, gold_targets):
        # if we have multiple gold words, we'll compare the top 2 words we predicted
        # to the actual top 2 words. if there are less gold words, then we'll just
        # look at our top gold prediction
        gold_lst = [i.strip() for i in gold.split(',')]
        for p, g in zip(pred, gold_lst):
            correct += p == g
            total += 1
    print("Accuracy: {:.2f}".format(correct / total * 100))

    print(f"Finished in {(time.time() - start) / 60} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--gen_dir', nargs='?', type=str, help='Directory to save predictions to', default='results')
    parser.add_argument('--split', nargs='?', type=str, help='Subset of data to consider', default='val')
    parser.add_argument('--n_samples', nargs='?', type=int, help='Number of samples', default=None)
    args = parser.parse_args()

    fire.Fire(main)