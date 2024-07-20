import numpy as np
import torch
import torch.nn.functional as F


## FN DEFINITIONS

def cosine_similarity(
    clue_embed: torch.Tensor,
    guess_embed: torch.Tensor,
    not_guess_embed: torch.Tensor,
) -> torch.Tensor:
    """
    clue_embed: (1, num_tokens, embedding_dim)
    guess_embed: (batch_size, num_tokens, embedding_dim)
    not_guess_embed: (batch_size, num_tokens, embedding_dim)
    
    return: loss
    """
    clue_guess_sim = pairwise_cosine_similarity(clue_embed, guess_embed)
    clue_not_guess_sim = pairwise_cosine_similarity(clue_embed, not_guess_embed)
    loss =  torch.mean(clue_not_guess_sim) - torch.mean(clue_guess_sim)
    return loss


def l2(
    clue_embed: torch.Tensor,
    guess_embed: torch.Tensor,
    not_guess_embed: torch.Tensor,
) -> torch.Tensor:
    """
    clue_embed: (1, num_tokens, embedding_dim)
    guess_embed: (batch_size, num_tokens, embedding_dim)
    not_guess_embed: (batch_size, num_tokens, embedding_dim)
    
    return: loss
    """
    clue_guess_dist = torch.cdist(clue_embed, guess_embed)
    clue_not_guess_dist = torch.cdist(clue_embed, not_guess_embed)
    loss = torch.mean(clue_guess_dist) - torch.mean(clue_not_guess_dist)
    return loss


def clip(
    clue_embed: torch.Tensor,
    guess_embed: torch.Tensor,
    not_guess_embed: torch.Tensor, 
) -> torch.Tensor:
    """
    clue_embed: (1, num_tokens, embedding_dim)
    guess_embed: (batch_size, num_tokens, embedding_dim)
    not_guess_embed: (batch_size, num_tokens, embedding_dim)
    
    return: loss
    """
    device = clue_embed.device

    clue_embed = F.normalize(clue_embed, dim=2).mean(dim=1)
    guess_embed = F.normalize(guess_embed, dim=2).mean(dim=1)
    not_guess_embed = F.normalize(not_guess_embed, dim=2).mean(dim=1)

    # TODO: maybe learn temperature
    t = 0.07
    guess_not_guess = torch.concatenate((guess_embed, not_guess_embed))
    logits = torch.matmul(clue_embed, guess_not_guess.T) * np.exp(t)

    guess_targets = torch.ones((1, guess_embed.shape[0])).to(device)
    guess_targets = F.normalize(guess_targets, p=1)
    not_guess_targets = torch.zeros((1, not_guess_embed.shape[0])).to(device)
    targets = torch.concatenate((guess_targets, not_guess_targets), dim=1)

    return F.cross_entropy(logits, targets)

def clip_giver(
    clue_embed: torch.Tensor,
    targets_embed: torch.Tensor,
    neutral_embed: torch.Tensor,
    avoid_embed: torch.Tensor,
) -> torch.Tensor:
    """
    clue_embed: (1, num_tokens, embedding_dim)
    targets_embed: (batch_size, num_tokens, embedding_dim)
    goal_embed: (batch_size, num_tokens, embedding_dim)
    neutral_embed: (batch_size, num_tokens, embedding_dim)
    avoid_embed: (batch_size, num_tokens, embedding_dim)
    
    return: loss
    """
    device = clue_embed.device

    clue_embed = F.normalize(clue_embed, dim=2).mean(dim=1)
    targets_embed = F.normalize(targets_embed, dim=2).mean(dim=1)
    neutral_embed = F.normalize(neutral_embed, dim=2).mean(dim=1)
    avoid_embed = F.normalize(avoid_embed, dim=2).mean(dim=1)

    # TODO: maybe learn temperature
    t = 0.07
    all_words = torch.concatenate((targets_embed, neutral_embed, avoid_embed))
    logits = torch.matmul(clue_embed, all_words.T) * np.exp(t)

    target_targets = torch.ones((1, targets_embed.shape[0])).to(device)
    target_targets = F.normalize(target_targets, p=1)
    neutral_targets = torch.zeros((1, neutral_embed.shape[0])).to(device)
    avoid_targets = torch.zeros((1, avoid_embed.shape[0])).to(device)
    targets = torch.concatenate((target_targets, neutral_targets, avoid_targets), dim=1)

    return F.cross_entropy(logits, targets)


loss_fns = {
    "cosine_similarity": cosine_similarity,
    "l2": l2,
    "clip": clip,
    "clip_giver": clip_giver,
}

## UTILS

def pairwise_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    a: (batch_a, num_tokens, embedding_dim)
    b: (batch_b, num_tokens, embedding_dim)

    return: (batch_a, batch_b)
    """
    # average across tokens and expand dimensions for matrix magic
    a = a.mean(dim=1).unsqueeze(1)    # (batch_a, 1, embedding_dim)
    b = b.mean(dim=1).unsqueeze(0)    # (1, batch_b, embedding_dim)
    return F.cosine_similarity(a, b, dim, eps)