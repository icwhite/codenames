from typing import List, Sequence
import torch
from codenames.players import BaseGiver
from codenames.embeddings import BertEmbeddings, GuesserEmbeddingDataset
import torch.nn.functional as F
from IPython import embed
import time
import itertools
from codenames.utils.replay_buffer import ReplayBuffer
import numpy as np


class SimilaritiesGiver(BaseGiver):

    def __init__(
        self, 
        embeddings: BertEmbeddings, 
        adaptive: bool = False,
        choose_argmax: bool = True,
        batch_size: int = 64, 
        buffer_size: int = 10000, 
        k: int = 3,
        max_num_targets: int = 1,
        tau: float = 0.0,
        neutral_penalty: float = 0.5,
        random_target_selection: bool = True,
    ):
        super().__init__()
        self.k = k
        self.adaptive = adaptive
        self.embeddings = embeddings
        self.choose_argmax = choose_argmax
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.num_observations = 0
        self.max_num_targets = max_num_targets
        self.tau = tau
        self.neutral_penalty = neutral_penalty
        self.random_target_selection = random_target_selection
        # self.train_clues, self.train_guesses, self.train_not_guesses = [], [], []
    
    def clue_similarities(
        self, 
        targets: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
    ) -> Sequence[float]:
        clue_embedding = self.embeddings.encode_and_embed(clues) # shape: (num_possible_clues, num_tokens, embedding_dim)
        target_embedding = self.embeddings.encode_and_embed(targets) # shape: (num_targets, num_tokens, embedding_dim)

        num_clues, num_tokens, embedding_dim = clue_embedding.size()

        cos = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
        # embed()
        similarities = torch.abs(cos(clue_embedding.unsqueeze(1), target_embedding.unsqueeze(0)))
        assert similarities.size() == (clue_embedding.shape[0], target_embedding.shape[0], num_tokens)
        similarities = torch.mean(similarities, dim=2) # take average of tokens
        total_similarities = torch.mean(similarities, dim=1)
        # total_similarities = torch.mean(similarities, dim=1)
        assert total_similarities.size() == (clue_embedding.shape[0],)
        return total_similarities
    
    def clue_similarities_batch(
        self, 
        targets: Sequence[Sequence[str]],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
    ) -> Sequence[float]:
        target_embedding = self.embeddings.encode_and_embed(targets)
        clue_embedding = self.embeddings.encode_and_embed(clues)

        num_clues, num_clue_tokens, embedding_dim = clue_embedding.size()

        cos = torch.nn.CosineSimilarity(dim=4, eps=1e-6)
        similarities = torch.abs(cos(clue_embedding.unsqueeze(1).unsqueeze(3), target_embedding.unsqueeze(0).unsqueeze(2)))
        assert similarities.size() == (num_clues, len(targets), num_clue_tokens, 1)
        similarities = torch.mean(similarities, dim=3).mean(dim=2)
        return similarities
    
    def select_targets(
        self, 
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
        num_targets: int = 1,
    ):
        if self.random_target_selection:
            return np.random.choice(goal, size=num_targets, replace=False)
        possible_targets = self._pick_possible_targets(goal, num_targets, True)
        total_score = self.clue_similarities_batch(possible_targets, avoid, neutral, clues)
        best_clue_probabilities, _ = torch.max(total_score, dim=0)
        best_idx = torch.argmax(best_clue_probabilities)
        best_target = possible_targets[best_idx.item()]
        return best_target
    
    def _pick_possible_targets(self, 
                               goal: Sequence[str],
                               max_num_targets: int, 
                               pick_exact: bool = False) -> List[str]:
        if len(goal) < max_num_targets:
            return [goal]
        if pick_exact: 
            possible_targets = [[goal[i] for i in idx] for idx in itertools.combinations(range(len(goal)), max_num_targets)]
            return possible_targets
        possible_targets = []
        for k in range(1, min(len(goal), max_num_targets + 1)):
            possible_targets.extend([[goal[i] for i in idx] for idx in itertools.combinations(range(len(goal)), k)])
        return possible_targets

    def give_clue(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
        targets: Sequence[str],
        choose_argmax: bool = False,
    ) -> str:
        total_similarities = self.clue_similarities(targets, avoid, neutral, clues)
        idx = torch.argmax(total_similarities)
        return clues[idx]
    
    def observe_turn(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
        not_guess: Sequence[str],
    ):
        pass
