import fire
import time
import argparse
import json
import pandas as pd

from llama_exps.clue_selection import generate_clues

from llama_exps.utils import load_model_and_tokenizer, get_fixed_generation_prob

def main(model_name: str,
         gen_dir: str = 'results', 
         split: str = 'val',
         k: int = 5,
         n_samples: int = None):
    
    assert split in ['all', 'test', 'train', 'val']
    start = time.time()

    # NOTE use chat!
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # # Take goal/avoid/neutral words => have LlamaGiver generate k clues
    # # (note that we may have repeats)
    # df = None
    # for i in range(k):
    #     # empty demographic
    #     clue_df = generate_clues(model=model, tokenizer=tokenizer, demo='', split=split, save_df=False, logprobs=True, n_samples=n_samples)
    #     if i == 0:
    #         # keep the other columns as well, which is why we dup the clue_df rather than just tack on the clue/lp columns 
    #         df = clue_df.rename(columns={'clues': 'clue_0', 'logprobs': 'lps_0'})
    #     else:
    #         df[f'clue_{i}'] = clue_df['clues']
    #         df[f'lps_{i}'] = clue_df['logprobs']

    # df.to_csv('results/intermediate_rsa_clues.csv')
    # print("Finished generating clues.")

    df = pd.read_csv('results/intermediate_rsa_clues.csv')

    # Take goal/avoid/neutral + clue => have LlamaGuesser identify guesses (2)
    gold_df = pd.read_csv(f'cultural-codes/codenames/data/clue_generation_task/{split}.csv')
    if n_samples:
        gold_df = gold_df[:n_samples]

    gold_targets = gold_df['base_text'].apply(lambda x: [i.strip() for i in x.split('targets: [')[1].split(']')[0].replace("'", "").split(',')]).tolist()
    num_targets_per_question = [len(i) for i in gold_targets]

    data = gold_df['base_text'].tolist()

    prompts, word_opts = [], []
    for i, d in enumerate(data):
        card_types = ['black', 'tan', 'targets']
        all_words = []
        for card_type in card_types:
            words = d.split(f"{card_type}: [")[1].split("]")[0].replace("'", "").split(",")
            words = [w.strip() for w in words]
            all_words.extend(words)

        word_opts.append(all_words)
        all_words = ", ".join(all_words)
        prompt = f"You are playing Codenames and are the clue guesser. You need to select one word from {all_words}."
        prompts.append(prompt)

    reranking = {k: [] for k in range(n_samples if n_samples else len(data))} # store question idx as key
    # we go through the ith clue for each prompt
    for clue_idx in range(k):
        print(f"starting {clue_idx}/{k}")
        clues = df[f'clue_{clue_idx}'].tolist()
        clue_lps = df[f'lps_{clue_idx}'].tolist()

        repeated_prompts, repeated_responses = [], []
        for p, w, clue in zip(prompts, word_opts, clues):
            p_w_clue = f"{p} Given the clue {clue}, the most likely word is "
            repeated_prompts.extend([p_w_clue] * len(w))
            repeated_responses.extend(w)

        output_lps = get_fixed_generation_prob(model,
                           tokenizer,
                           sequences=repeated_prompts, 
                           responses=repeated_responses,
                           batch_size=2) # FIXME for llama3 constraints

        for q_idx, w in enumerate(word_opts): # len(word_opts) == n_samples, i.e. number of questions / unrepeated prompts
            num_words = len(w)
            curr_lps, output_lps = output_lps[:num_words], output_lps[num_words:]

            # add the clue lp directly here
            curr_lps = [w_lp + clue_lps[q_idx] for w_lp in curr_lps]

            # get the most likely `n` words for each data point, corresponding to the `n` gold
            # targets there are
            num_targets = num_targets_per_question[q_idx]
            zipped = list(zip(curr_lps, w))
            zipped.sort(key = lambda x: -x[0]) # sort by lp, descending

            # store the top `n` words for the question. later we'll limit to the actual top `n` across clue indices
            top_n_zipped_w_clue = [(z[0], z[1], clues[q_idx]) for z in zipped[:num_targets]] # tuple are (total clue + word lp, guessed word, clue)
            reranking[q_idx].extend(top_n_zipped_w_clue)
        
        json.dump(reranking, open(f'{gen_dir}/reranking_{clue_idx}.json', 'w'))

    # for each question, find the top `n` unique word selections
    top_reranking = {}
    for q_idx, opts in reranking.items():
        num_targets = num_targets_per_question[q_idx]
        opts.sort(key = lambda x: -x[0])

        top_n_opts = []
        seen_words = []

        # naively iterate through opts until we find enough unique word guesses
        while len(top_n_opts) < num_targets:
            opt = opts.pop(0)
            guess_word = opt[1]
            if guess_word not in seen_words:
                top_n_opts.append(opt)
                seen_words.append(guess_word)

        top_reranking[q_idx] = top_n_opts

    # store results
    gold_clues = gold_df['output'].tolist()

    output_guesses, output_clues, output_lps = [], [], []
    for v in top_reranking.values():
        # bc we may have multiple targets, need to do another list unpacking
        output_lps.append([i[0] for i in v])
        output_guesses.append([i[1] for i in v])
        output_clues.append([i[2] for i in v])
        
    output_df = pd.DataFrame({
        'clue': output_clues,
        'guess': output_guesses,
        'total logprob': output_lps,
        'gold clues': gold_clues,
        'gold targets': gold_targets,
    })
    output_df.to_csv(f'{gen_dir}/llama_rsa_{n_samples}.csv', index=False)
    
    # compute accuracy by comparing the most likely guess to the human target(s) (note
    # that there may be multiple targets)
    correct_matches = 0
    total_targets = sum(num_targets_per_question)
    for gold, preds in zip(gold_targets, output_guesses):
        for p in preds:
            correct_matches += p in gold
    acc = correct_matches / total_targets

    print(f"Target accuracy: {acc * 100 :.2f}%")

    # compute clue accuracy post reranking
    total, correct = 0, 0
    for p, g in zip(output_clues, gold_clues):
        g_opts = [i.strip() for i in g.split(',')]
        for p_opt, g_opt in zip(p, g_opts):
            total += 1
            correct += p_opt == g_opt
            
    print(f"Clue accuracy: {correct / total * 100 :.2f}%")

    print(f"Finished in {(time.time() - start) / 60} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--gen_dir', nargs='?', type=str, help='Directory to save predictions to', default='results')
    parser.add_argument('--split', nargs='?', type=str, help='Subset of data to consider', default='val')
    parser.add_argument('--n_samples', nargs='?', type=int, help='Number of samples of data', default=None)
    args = parser.parse_args()

    fire.Fire(main)