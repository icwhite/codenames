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


class LiteralGiver(BaseGiver):

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
        avoid_embedding = self.embeddings.encode_and_embed(avoid) # shape: (num_avoid, num_tokens, embedding_dim)
        neutral_embedding = self.embeddings.encode_and_embed(neutral) # shape: (num_neutral, num_tokens, embedding_dim)

        num_clues, num_tokens, embedding_dim = clue_embedding.size()
        num_avoid, num_avoid_tokens, _ = avoid_embedding.size()
        num_neutral, num_neutral_tokens, _ = neutral_embedding.size()

        cos = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
        # embed()
        similarities = torch.abs(cos(clue_embedding.unsqueeze(1), target_embedding.unsqueeze(0)))
        assert similarities.size() == (clue_embedding.shape[0], target_embedding.shape[0], num_tokens)
        similarities = torch.mean(similarities, dim=2) # take average of tokens
        avoid_similarities = torch.abs(torch.nn.functional.cosine_similarity(clue_embedding.unsqueeze(1).unsqueeze(3), 
                                                                   avoid_embedding.unsqueeze(0).unsqueeze(2),
                                                                   dim=-1))
        neutral_similarities = torch.abs(torch.nn.functional.cosine_similarity(clue_embedding.unsqueeze(1).unsqueeze(3), 
                                                                   neutral_embedding.unsqueeze(0).unsqueeze(2),
                                                                   dim=-1))
        assert avoid_similarities.size() == (clue_embedding.shape[0], avoid_embedding.shape[0], num_tokens, num_avoid_tokens)
        assert neutral_similarities.size() == (clue_embedding.shape[0], neutral_embedding.shape[0], num_tokens, num_neutral_tokens)
        avoid_similarities = torch.mean(avoid_similarities, dim=3).mean(dim=2) # take average of tokens
        neutral_similarities = torch.mean(neutral_similarities, dim=3).mean(dim=2) # take average of tokens
        total_similarities = torch.mean(similarities, dim=1) - torch.mean(avoid_similarities, dim=1) - 0.5*torch.mean(neutral_similarities, dim=1)
        # total_similarities = torch.mean(similarities, dim=1)
        assert total_similarities.size() == (clue_embedding.shape[0],)
        return total_similarities
    
    def guess_probabilties_batch(self, 
                                 unselected: Sequence[str],
                                 clue: Sequence[str]) -> Sequence[float]:
        # Encode and embed the clue
        clue_embedding = self.embeddings.encode_and_embed(clue) # shape: (num_clues, num_tokens, embedding_dim)
        num_clues, num_clue_tokens, embedding_dim = clue_embedding.size()
        # Encode and embed the words
        word_embeddings = self.embeddings.encode_and_embed(unselected)
        num_words, num_word_tokens, embedding_dim = word_embeddings.size()

        # Calculate the cosine similarity between the clue and the words
        cosine_similarities = torch.abs(torch.nn.functional.cosine_similarity(clue_embedding.unsqueeze(1).unsqueeze(3), 
                                                                   word_embeddings.unsqueeze(0).unsqueeze(2),
                                                                   dim=-1))
        # Convert the similarity to probabilities
        assert cosine_similarities.size() == (num_clues, num_words, num_clue_tokens, num_word_tokens)
        cosine_similarities = torch.mean(cosine_similarities, dim=3).mean(dim=2)
        probabilities = F.softmax(cosine_similarities, dim=1)
        assert probabilities.size() == (num_clues, num_words)
        return probabilities
    
    def clue_probabilities(
        self, 
        targets: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
    ):
        unselected = targets + avoid + neutral
        guess_probabilities = self.guess_probabilties_batch(unselected, clues) #shape: (num_clues, num_unselected)
        target_idx = [unselected.index(target) for target in targets]
        avoid_idx = [unselected.index(avoid_word) for avoid_word in avoid]
        neutral_idx = [unselected.index(neutral_word) for neutral_word in neutral]
        target_probabilities = guess_probabilities[:, target_idx]
        avoid_probabilities = guess_probabilities[:, avoid_idx]
        neutral_probabilities = guess_probabilities[:, neutral_idx]
        # total_probabilities 
        total_probabilities = torch.mean(target_probabilities, dim=1) - torch.max(avoid_probabilities, dim=1).values - 0.5*torch.max(neutral_probabilities, dim=1).values 
        return total_probabilities
    
    def clue_probabilities_batch(
        self, 
        goal: Sequence[str],
        targets: Sequence[Sequence[str]],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
        neutral_penalty: float = 0.5,
    ):
        """
        goal: list of goal words
        targets: list of list of target words where each list of target words has the same length
        avoid: list of avoid words
        neutral: list of neutral words
        clues: list of clue words
        """
        unselected = goal + avoid + neutral
        guess_probabilities = self.guess_probabilties_batch(unselected, clues) #shape: (num_clues, num_unselected)

        target_idx = [[unselected.index(target) for target in target_lst] for target_lst in targets]
        target_probabilities = torch.stack([guess_probabilities[:, idx] for idx in target_idx], dim=1) # shape: (num_clues, num_possible_targets, num_targets)
        avoid_idx = [unselected.index(avoid_word) for avoid_word in avoid]
        avoid_probabilities = guess_probabilities[:, avoid_idx]

        neutral_idx = [unselected.index(neutral_word) for neutral_word in neutral]
        neutral_probabilities = guess_probabilities[:, neutral_idx]
        total_score = torch.mean(target_probabilities, dim=2) -\
              torch.max(avoid_probabilities, dim=1).values.unsqueeze(dim=1)\
                  - self.neutral_penalty*torch.max(neutral_probabilities, dim=1).values.unsqueeze(dim=1) # shape (num_clues, num_possible_targets)
        diff_score = torch.min(target_probabilities, dim=2).values - torch.max(avoid_probabilities, dim=1).values.unsqueeze(dim=1) # shape (num_clues, num_possible_targets)
        return total_score, diff_score
    
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

    def select_targets(
        self, 
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
        num_targets: int = 2,
    ) -> Sequence[str]: 
        possible_targets = self._pick_possible_targets(goal, num_targets, True)
        total_score, _ = self.clue_probabilities_batch(goal, possible_targets, avoid, neutral, clues)
        best_clue_probabilities, _ = torch.max(total_score, dim=0) # shape: (num_possible_targets)
        best_idx = torch.argmax(best_clue_probabilities)
        best_target = possible_targets[best_idx.item()]
        return best_target
    
    # def select_targets(
    #     self, 
    #     goal: Sequence[str],
    #     avoid: Sequence[str],
    #     neutral: Sequence[str],
    #     clues: List[str],
    # ) -> Sequence[str]: 
        # max_num_targets = self.max_num_targets
        # if max_num_targets == 1:
        #     # select target randomly
        #     return [np.random.choice(goal)]

        # possible_targets = self._pick_possible_targets(goal, max_num_targets, True)
        # diff = -float('inf')
        # while diff < self.tau and self.max_num_targets >= 1:
        #     possible_targets = self._pick_possible_targets(goal, max_num_targets, True)
        #     total_score, difference_score = self.clue_probabilities_batch(goal, possible_targets, avoid, neutral, clues)
        #     best_clue_probabilities, indices = torch.max(total_score, dim=0) # shape: (num_possible_targets)
        #     best_idx = torch.argmax(best_clue_probabilities)
        #     diff = difference_score[indices[best_idx], best_idx]
        #     max_num_targets -= 1
        # best_target = possible_targets[best_idx.item()]
        # return best_target


    def give_clue(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        clues: List[str],
        targets: Sequence[str],
        choose_argmax: bool = False,
    ) -> str:
        total_score, diff_score = self.clue_probabilities_batch(goal, [targets], avoid, neutral, clues)
        idx = torch.argmax(total_score)
        return clues[idx]
    
    def observe_turn(
        self,
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
    ):
        pass
