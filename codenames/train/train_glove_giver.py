import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from codenames.embeddings import *
from codenames.eval.human_agreement import *
from codenames.players import LiteralGiver
from scripts.parse_cultural_codes import *


def main(
    desc: str,
    train_set: str = "train",
    val_set: str = "val",
    glove_dim: int = 300,
    embed_dim: int = 1000,
    lr: float = 1e-4,
):
    wandb.init(
        project="codenames",
        name=f"glove giver {desc}",
        config=vars(args),
    )

    print("Loading embeddings and dataset")
    embeddings = TrainGiverEmbeddings(
        in_embed_dim=glove_dim,
        out_embed_dim=embed_dim,
        loss_fn="clip_giver",
        lr=lr,
    )
    train_dataset = get_correct_clues_dataset(train_set)
    val_dataset = get_correct_clues_dataset(val_set)

    clue_df = parse_correct_clues("val")
    target_df = parse_correct_targets("val")

    for i in range(25):
        print(f"Epoch {i}")
        train_loss = embeddings.train(train_dataset, batch_size=32)
        val_loss = embeddings.val(val_dataset)
        log = {
            "training/epoch": i,
            "training/train_loss": train_loss,
            "training/val_loss": val_loss,
        }

        giver = LiteralGiver(embeddings)
        giver_agreement = (check_clue_agreement(giver, clue_df) | 
                        check_target_selection_agreement(giver, target_df))
        for k, v in giver_agreement.items():
            log[f"giver/{k}"] = v

        wandb.log(log)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, help="run description")
    parser.add_argument("--train_set", type=str, default="train", help="split of data used for training")
    parser.add_argument("--val_set", type=str, default="val", help="split of data used for val")
    parser.add_argument("--glove_dim", type=int, choices={50, 100, 200, 300}, default=300, help="GloVe embedding dimension")
    parser.add_argument("--embed_dim", type=int, default=1000, help="trained embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    main(**vars(args))