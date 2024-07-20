from typing import List
from codenames.players import PragmaticGiver
from codenames.players.literal_guesser import LiteralGuesser
from codenames.embeddings import *
import numpy as np 
import torch

class MatchGiver(PragmaticGiver):
    def __init__(
        self, 
        embeddings: TrainGuesserEmbeddings, 
        embeddings_list: List[TrainGuesserEmbeddings],
        alpha: float = 0.7,
        adaptive: bool = False,
        choose_argmax: bool = False,
        batch_size: int = 64, 
        buffer_size: int = 10000, 
        k:int = 23,
        max_num_targets: int = 3, 
        tau: float = 0.0, 
        neutral_penalty: float = 0.5,
    ):
        super().__init__(
                          embeddings, 
                          adaptive,
                          choose_argmax,
                          batch_size,
                          buffer_size,
                          k, 
                          max_num_targets, 
                          tau=tau, 
                          neutral_penalty=neutral_penalty)
        self.embeddings_list = embeddings_list
        self.prob_same_culture = [1] * len(embeddings_list)
        self.alpha = alpha
        self.num_times_chosen = [0] * len(embeddings_list)
    
    def compute_prob_same_culture_multiple_targets(
        self, 
        goal: Sequence[str],
        targets: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clue: str,
        guess: Sequence[str],
        guesser: LiteralGuesser,
    ):
        total_prob = 1
        unselected = list(set(goal) | set(avoid) | set(neutral))
        possible_targets = self._pick_possible_targets(unselected, self.max_num_targets, pick_exact=True)
        clue_probs = self.clue_probabilities_batch(possible_targets, goal, avoid, neutral, [clue])
        # 

        topk = torch.topk(guess_probs, len(guess)).indices
        topk_unselected = [unselected[i] for i in topk]
        for g in guess:
            if g in topk_unselected or g in targets: 
                continue
            g_idx = unselected.index(g)
            total_prob *= guess_probs[g_idx]
            # guess_probs = guesser.guess_probabilities(unselected, clue)
            # best_idx = np.argmax(guess_probs.numpy())
            # g_idx = unselected.index(g)
            # if best_idx == g_idx:
            #     continue
            # total_prob *= guess_probs[g_idx]
            # unselected.remove(g)
        assert total_prob > 0, total_prob < 1
        return total_prob
    
    
    def compute_prob_same_culture(
        self, 
        unselected: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
        guesser: LiteralGuesser,
    ):
        total_prob = 1
        guess_probs = guesser.guess_probabilities(unselected, clue)
        topk = torch.topk(guess_probs, len(guess)).indices
        topk_unselected = [unselected[i] for i in topk]
        for g in guess:
            if g in topk_unselected or g in targets: 
                continue
            g_idx = unselected.index(g)
            total_prob *= guess_probs[g_idx]
        assert total_prob > 0, total_prob < 1
        return total_prob

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
        
        for i, embedding in enumerate(self.embeddings_list):
            guesser = LiteralGuesser(embedding)
            unselected = list(set(goal) | set(avoid) | set(neutral))
            for g in guess:
                prob = self.compute_prob_same_culture(unselected, targets, clue, [g], guesser)
                unselected.remove(g)
            # self.prob_same_culture[i] = prob
                self.prob_same_culture[i] = self.prob_same_culture[i] * self.alpha + (1 - self.alpha) * prob
        
        best_idx = np.argmax(self.prob_same_culture)
        self.num_times_chosen[best_idx] += 1
        self.embeddings = self.embeddings_list[best_idx]
        self.literal_guesser = LiteralGuesser(self.embeddings, choose_argmax=self.choose_argmax)
        return {
            "probs": self.prob_same_culture,
        }