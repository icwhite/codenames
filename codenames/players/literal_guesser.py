import numpy as np
from typing import Optional, Sequence

from codenames.players import BaseGuesser
import random
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity


class LiteralGuesser(BaseGuesser):

    def __init__(self, 
                 embeddings, 
                 choose_argmax: Optional[bool] = False, 
                 adaptive: Optional[bool] = False,
                 ):
        super().__init__()
        # Set a random seed
        random_seed = 42
        random.seed(random_seed)

        self.choose_argmax = choose_argmax
        self.adaptive = adaptive
        # Set a random seed for PyTorch (for GPU as well)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        self.embeddings = embeddings

    def guess_probabilities(self, 
                            unselected: Sequence[str],
                            clue: str) -> Sequence[float]:
        # Encode and embed the clue
        clue_embedding = self.embeddings.encode_and_embed([clue]) # shape: (1, num_tokens, embedding_dim)
        num_clues, num_clue_tokens, embedding_dim = clue_embedding.size()
        # Encode and embed the words
        word_embeddings = self.embeddings.encode_and_embed(unselected) 
        num_words, num_word_tokens, embedding_dim = word_embeddings.size()
        
        # Calculate the cosine similarity between the clue and the words
        cos = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
        cosine_similarities = torch.abs(cos(clue_embedding.unsqueeze(2), word_embeddings.unsqueeze(1)))
        # Convert the similarity to probabilities
        assert cosine_similarities.size() == (num_words, num_clue_tokens, num_word_tokens)
        cosine_similarities = torch.mean(cosine_similarities, dim=2).mean(dim=1)
        probabilities = F.softmax(cosine_similarities, dim=0) # check dimensions
        assert probabilities.size() == (word_embeddings.size(0), )
        return probabilities
    
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
    
    def guess_clue_similarities(self,
                                clue: Sequence[str],
                                guess: Sequence[str]) -> Sequence[float]:
        # Encode and embed the clue
        clue_embedding = self.embeddings.encode_and_embed(clue) # shape: (num_clues, num_tokens, embedding_dim)
        num_clues, num_clue_tokens, embedding_dim = clue_embedding.size()
        # Encode and embed the words
        word_embeddings = self.embeddings.encode_and_embed(guess)
        num_words, num_word_tokens, embedding_dim = word_embeddings.size()
        # print(clue_embedding.size(), word_embeddings.size())
        # Calculate the cosine similarity between the clue and the words
        cosine_similarities = torch.abs(torch.nn.functional.cosine_similarity(clue_embedding.unsqueeze(1).unsqueeze(3), 
                                                                   word_embeddings.unsqueeze(0).unsqueeze(2),
                                                                   dim=-1))
        # Convert the similarity to probabilities
        assert cosine_similarities.size() == (num_clues, num_words, num_clue_tokens, num_word_tokens)
        cosine_similarity = torch.mean(cosine_similarities, dim=3).mean(dim=2).mean(dim=1).item()
        # print(cosine_similarities.size())
        return cosine_similarity

    def make_guess(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int = 1,
        choose_argmax: Optional[bool] = False,
    ) -> Sequence[str]:
        probabilities = self.guess_probabilities(unselected, clue)
        if choose_argmax or self.choose_argmax:
            return np.array(unselected)[np.argsort(probabilities.numpy())[-num_targets:]]
        # select the words with the highest probabilities
        return np.random.choice(unselected, num_targets, replace=False, p=probabilities.numpy().flatten())


    def observe_turn(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
        guess: Sequence[str],
        result: Sequence[str],
    ):
        return {}
        