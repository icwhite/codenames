import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from codenames.embeddings import *
from codenames.eval.human_agreement import *
from codenames.players import LiteralGuesser, PragmaticGiver
from codenames.utils.multi_var_splits import *
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
    train_set: str = "train",
    val_set: str = "val",
    glove_dim: int = 300,
    embed_dim: int = 1000,
    loss: str = "clip",
    lr: float = 1e-4,
    save: bool = False,
):
    wandb.init(
        project="codenames",
        name=f"glove {group.lower()}",
        config=vars(args),
    )

    print("Loading embeddings and dataset")
    embeddings = TrainGuesserEmbeddings(
        in_embed_dim=glove_dim,
        out_embed_dim=embed_dim,
        loss_fn=loss,
        lr=lr,
    )

    group_list = [x.strip() for x in group.split(',')]
    train_dataset = get_generate_guess_dataset_culture_splits(train_set, group_list)
    val_dataset = get_generate_guess_dataset_culture_splits(val_set, group_list)

    df = parse_sub_levels('val')
    df.dropna(how="any", inplace=True)
    for g in group_list:
        df = df[make_filter(df, g)]

    log = {"training/epoch": 0}
    guesser = LiteralGuesser(embeddings)
    guesser_agreement = check_agreement_new(guesser, df)
    for k, v in guesser_agreement.items():
        log[f"guesser/{k}"] = v

    wandb.log(log)

    for i in range(1, 26):
        print(f"Epoch {i}")
        train_loss = embeddings.train(train_dataset, batch_size=32)
        val_loss = embeddings.val(val_dataset)
        log = {
            "training/epoch": i,
            "training/train_loss": train_loss,
            "training/val_loss": val_loss,
        }

        guesser = LiteralGuesser(embeddings)
        guesser_agreement = check_agreement_new(guesser, df)
        for k, v in guesser_agreement.items():
            log[f"guesser/{k}"] = v

        wandb.log(log)

    if save:
        print("Saving weights")
        os.makedirs("models", exist_ok=True)
        torch.save(
            embeddings.trainable,
            os.path.join("models", f"{group.lower().replace(' ', '_')}_{glove_dim}_{embed_dim}.pth"))

    wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, default="train", help="split of data used for training")
    parser.add_argument("--val_set", type=str, default="val", help="split of data used for val")
    parser.add_argument("--glove_dim", type=int, choices={50, 100, 200, 300}, default=300, help="GloVe embedding dimension")
    parser.add_argument("--embed_dim", type=int, default=1000, help="trained embedding dimension")
    parser.add_argument("--loss", type=str, choices={"cosine_similarity", "l2", "clip"}, default="clip", help="loss function to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--group", type=str, required=True, help="culture split group(s) to train on")
    parser.add_argument("--save", action="store_true", help="save weights after training")
    args = parser.parse_args()

    arg_dict = vars(args)
    if args.group == "all":
        for group in ALL_GROUPS:
            arg_dict["group"] = group
            main(**arg_dict)
    elif args.group == "all_two_splits":
        for group in get_all_two_splits()[18:]:
            arg_dict["group"] = ", ".join(group)
            main(**arg_dict)
    else:
        main(**arg_dict)