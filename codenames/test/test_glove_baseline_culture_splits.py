import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from codenames.embeddings import *
from codenames.eval.human_agreement import *
from codenames.players import LiteralGuesser, PragmaticGiver
from scripts.parse_cultural_codes import *

    
ALL_GROUPS = [
    "AGE under 30",
    "AGE over 30",
    "NATIVE true",
    "NATIVE false",
    "COUNTRY united states",
    "COUNTRY foreign",
    "GENDER male",
    "GENDER female",
    "EDUCATION high school associate",
    "EDUCATION bachelor",
    "EDUCATION graduate",
    "RELIGION catholic",
    "RELIGION not catholic",
    "POLITICAL liberal",
    "POLITICAL conservative",
]


def main(
    group: str,
    glove_dim: int = 300,
    embeddings = None,
):
    wandb.init(
        project="codenames",
        name=f"glove baseline {group.lower()}",
        config=vars(args),
    )

    print("Loading embeddings and dataset")
    if embeddings is None:
        embeddings = GloveEmbeddings(embed_dim=glove_dim)

    group_list = [x.strip() for x in group.split(',')]
    df = parse_sub_levels('val')
    df.dropna(how="any", inplace=True)
    for g in group_list:
        df = df[make_filter(df, g)]

    for i in range(25):

        log = {"training/epoch": i}
        guesser = LiteralGuesser(embeddings)
        guesser_agreement = check_agreement_new(guesser, df)
        for k, v in guesser_agreement.items():
            log[f"guesser/{k}"] = v

        wandb.log(log)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_dim", type=int, choices={50, 100, 200, 300}, default=300, help="GloVe embedding dimension")
    parser.add_argument("--group", type=str, required=True, help="culture split group(s) to train on")
    args = parser.parse_args()

    arg_dict = vars(args)

    # if True:
    #     embeddings = TrainEmbeddings(
    #         in_embed_dim=300,
    #         out_embed_dim=1000,
    #         loss_fn="clip",
    #         lr=1e-4,
    #     )
    #     train_dataset = get_generate_guess_dataset("train")
    #     for i in range(25):
    #         embeddings.train(train_dataset, batch_size=32)
    #     arg_dict["embeddings"] = embeddings

    if args.group == "all":
        for group in ALL_GROUPS:
            arg_dict["group"] = group
            main(**arg_dict)
    else:
        main(**arg_dict)