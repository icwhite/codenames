from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .embedding_dataset import GuesserEmbeddingDataset
from .loss_fns import loss_fns
    
class Embeddings(ABC):
    #TODO: refactor such that no longer dependent on BERT tokenizer and model paradigm and more flexible
    def __init__(
        self,
        loss_fn: str = "cosine_similarity",
    ):
        self.embedding_loss = loss_fns[loss_fn]
        
    def encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            return self._encode_and_embed(texts).cpu()
    
    @abstractmethod
    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        ...
    
    def similarity_to_probability(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        """
        similarity: torch.Tensor, shape (num_clues, num_words)
        """
        return F.softmax(similarity, dim=1)

    def copy(self):
        return self.__class__()