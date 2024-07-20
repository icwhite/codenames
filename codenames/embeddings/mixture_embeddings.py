from typing import Sequence
from .word_embeddings import Embeddings
import torch

class MixtureEmbeddings(Embeddings):
    def __init__(self, 
                 embeddings1: Embeddings,
                 embeddings2: Embeddings,
                 mixture: float = 0.5,
                 ):
        super().__init__()
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.mixture = mixture
    
    def update_mixture(self, mixture: float):
        self.mixture = mixture
    
    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        """
        with torch.no_grad():
            base_embedding1 = self.embeddings1._encode_and_embed(texts)
            base_embedding2 = self.embeddings2._encode_and_embed(texts)
        return self.mixture * base_embedding1 + (1 - self.mixture) * base_embedding2