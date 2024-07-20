import argparse
import pandas as pd
import time

from codenames.embeddings.glove_embeddings import GloveEmbeddings, TrainGuesserEmbeddings
from codenames.players.literal_guesser import LiteralGuesser
from codenames.players.llama_guesser import LlamaGuesser
from codenames.players.match_giver import MatchGiver
from codenames.players.similarities_giver import SimilaritiesGiver
from codenames.eval.interactive_evaluation import play_game_eval

def run_match(guesser, 
              embeddings, 
              embeddings_lst, 
              num_games, 
              verbose):
    """
    Returns results of running the game eval.
    """
    match_giver = MatchGiver(embeddings,
                    embeddings_lst, 
                    choose_argmax=True, 
                    alpha=0.3, 
                    max_num_targets=args.max_num_targets, 
                    tau=args.tau,
                    neutral_penalty=args.neutral_penalty)
    results = play_game_eval(guesser, match_giver, num_games=num_games, verbose=verbose)
    print(match_giver.num_times_chosen)
    return results

def run_similarities(guesser, 
                     embeddings, 
                     num_games, 
                     verbose):
    similarities_giver = SimilaritiesGiver(embeddings)
    results = play_game_eval(guesser, similarities_giver, num_games=num_games, verbose=verbose)
    return results

def test_models(models, 
                model_names, 
                guesser, 
                guesser_name,
                num_games,
                verbose):
    
    start_time = time.time()
    results = []
    
    test_embeddings = []
    for model in models:
        embeddings = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        embeddings.load_weights(model)
        test_embeddings.append(embeddings)

    if args.do_similarities:
        print("running literal giver")
        for embedding, model_name in zip(test_embeddings, model_names):
            match = run_similarities(guesser, embedding, num_games, verbose)
            results.append({
                "guesser": guesser_name,
                "giver": model_name + "_sim",
                **match
            })
    
    if args.do_individual:
        print("running pragmatic giver")
        for embedding, model_name in zip(test_embeddings, model_names):
            match = run_match(guesser, embedding, [embedding], num_games, verbose)
            results.append({
                "guesser": guesser_name,
                "giver": model_name + "_rsa",
                **match
            })

    print(f"first batch: {(time.time() - start_time) / 60} minutes")

    if args.do_match:
        print("running rsa_c3 giver")
        match = run_match(guesser, test_embeddings[0], test_embeddings, num_games, verbose)
        results.append({
            "guesser": guesser_name,
            "giver": " ".join(model_names),
            **match
        })

        print(f"second batch: {(time.time() - start_time) / 60} minutes")

        if not args.country_models:
            model_names = model_names[1:]
            match = run_match(guesser, test_embeddings[1], test_embeddings[1:], num_games, verbose)
            results.append({
                "guesser": guesser_name,
                "giver": " ".join(model_names),
                **match
            })

    print(f"third batch: {(time.time() - start_time) / 60} minutes")

    df = pd.DataFrame(results)

    gn = guesser_name.replace(' ', '_')
    name = "sim_" if args.do_similarities else ""
    name += "ind_" if args.do_individual else ""
    name += "match_" if args.do_match else ""
    fn = f"results/interactive_eval_{gn}_{name}_{num_games}_{args.neutral_penalty}_{args.max_num_targets}.csv"

    df.to_csv(fn)

    print(f"saved to {fn} in {(time.time() - start_time) / 60} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1, help="number of games to play")
    parser.add_argument("--max_num_targets", type=int, default=1, help="number of targets to guess")
    parser.add_argument("--llama_guesser", type=str, default="", help="model name for llama") 
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, help="print results")   
    parser.add_argument("--new_models", action=argparse.BooleanOptionalAction, help="save results to csv")
    parser.add_argument("--tau", type=float, default=0.0, help="threshold")
    parser.add_argument("--neutral_penalty", type=float, default=0.1, help="penalty for neutral words")
    parser.add_argument("--do_individual", action=argparse.BooleanOptionalAction, help="run individual matchups")
    parser.add_argument("--do_similarities", action=argparse.BooleanOptionalAction, help="run similarities")
    parser.add_argument("--do_match", action=argparse.BooleanOptionalAction, help="run match")
    parser.add_argument("--country_models", action=argparse.BooleanOptionalAction, help="run country models")
    args = parser.parse_args()

    embeddings = GloveEmbeddings(embed_dim=300)



    if args.new_models:
        models = [
            "splits_models/education_bachelor,_political_conservative_300_1000.pth",
            "splits_models/education_bachelor,_political_liberal_300_1000.pth",
            "splits_models/education_graduate,_political_conservative_300_1000.pth",
            "splits_models/education_graduate,_political_liberal_300_1000.pth",
            "splits_models/education_high_school_associate,_political_conservative_300_1000.pth",
            "splits_models/education_high_school_associate,_political_liberal_300_1000.pth",
        ]

        model_names = [
            "Bachelor Conservative",
            "Bachelor Liberal",
            "Graduate Conservative",
            "Graduate Liberal",
            "High School Associate Conservative",
            "High School Associate Liberal",
        ]
    elif args.country_models:
        models = ["models/country_united_states_300_1000.pth", "models/country_foreign_300_1000.pth"]
        model_names = ["United States", "Foreign"]
    else: 
        models = ["models/education_high_school_associate_300_1000.pth", 
              "models/education_graduate_300_1000.pth", 
              "models/education_bachelor_300_1000.pth",]
        model_names = ["High School/Associate", "Graduate", "Bachelor"]
    
    if args.llama_guesser: 
        guesser = LlamaGuesser(args.llama_guesser)
        guesser_name = f"llama-{args.llama_guesser}"

    else:
        # our guesser is HS/Associate by default
        guesser_embeddings = TrainGuesserEmbeddings(
            in_embed_dim=300,
            out_embed_dim=1000,
        )
        guesser_embeddings.load_weights(models[0])
        if args.new_models:
            guesser_name = "Bachelor Conservative"
        elif args.country_models:
            guesser_name = "domestic"
        else:
            guesser_name = "HS"
        guesser = LiteralGuesser(guesser_embeddings, choose_argmax=True)
    
    print(f"Running experiments for {guesser_name} guesser")
    test_models(models, 
                model_names, 
                guesser, 
                guesser_name, 
                num_games=args.num_games,
                verbose=args.verbose)
