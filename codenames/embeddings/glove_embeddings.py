from typing import Dict, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from tqdm import tqdm

from .embedding_dataset import GiverEmbeddingDataset, GuesserEmbeddingDataset
from .word_embeddings import Embeddings
from IPython import embed

class GloveEmbeddings(Embeddings):

    def __init__(self, 
                 embed_dim: int = 50,
                 loss_fn: str = "cosine_similarity",
                 ):
        super().__init__(loss_fn=loss_fn)
        self.embed_dim = embed_dim
        self.glove = GloVe(name='6B', dim=embed_dim)
        glove_weights = torch.load(f".vector_cache/glove.6B.{embed_dim}d.txt.pt")
        self.embedding = torch.nn.Embedding.from_pretrained(glove_weights[2])
    
    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        """

        tokenizer = get_tokenizer("basic_english")
        embeddings = []
        for item in texts:
            if type(item) == list:
                item = " ".join(item)
            indices = tokenizer(item)

            # tokenizer may return multiple tokens for a single word
            tok_embeddings = []
            for tok in indices:
                em = self.glove.get_vecs_by_tokens(tok)
                tok_embeddings.append(em.cpu())
            
            embedding = torch.mean(torch.stack(tok_embeddings), dim=0).unsqueeze(0)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        return embeddings


class TrainGuesserEmbeddings(GloveEmbeddings):

    def __init__(
        self, 
        in_embed_dim: int = 50, 
        out_embed_dim: int = 50,
        loss_fn: str = "cosine_similarity",
        lr: float = 1e-4,
    ):
        super().__init__(embed_dim=in_embed_dim, loss_fn=loss_fn)
        self.glove = GloVe(name='6B', dim=in_embed_dim)
        glove_weights = torch.load(f".vector_cache/glove.6B.{in_embed_dim}d.txt.pt")
        self.embedding = torch.nn.Embedding.from_pretrained(glove_weights[2])

        self.trainable = nn.Sequential(
            nn.Linear(in_embed_dim, out_embed_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainable.to(self.device)
        self.embedding.to(self.device)

        self.optim = torch.optim.Adam(self.trainable.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)

    def load_weights(self, path: str):
        self.trainable = torch.load(path)

    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        
        NOTE: if using multiple gpus, may have to additionally move `base_embedding` to one gpu, e.g. with: `base_embedding = base_embedding.to("cuda:0")` before
        passing into training. 
        """
        with torch.no_grad():
            base_embedding = super()._encode_and_embed(texts).to(self.device)
        return self.trainable(base_embedding)

    def train(
        self,
        dataset: GuesserEmbeddingDataset,
        batch_size: int = 1,
    ) -> float:
        join_words = dataset.join_words
        losses = []
        if batch_size == 1:
            dataset.join_words = False
            for clue, guess, not_guess in tqdm(dataset):
                loss = self.train_single(clue, guess, not_guess)
                losses.append(loss)
        else:
            dataset.join_words = True
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
            )
            for clue, guess, not_guess in tqdm(dataloader):
                loss = self.train_batch(clue, guess, not_guess) 
                losses.append(loss)
        self.scheduler.step()
        dataset.join_words = join_words

        return np.mean(losses)
    
    def val(
        self,
        dataset: GuesserEmbeddingDataset,
    ) -> float:
        losses = []
        for clue, guess, not_guess in tqdm(dataset):
            clue_embed = self.encode_and_embed([clue])
            guess_embed = self.encode_and_embed(guess)
            not_guess_embed = self.encode_and_embed(not_guess)

            loss = self.embedding_loss(clue_embed, guess_embed, not_guess_embed)
            losses.append(loss.item())

        return np.mean(losses)

    def train_single(
        self,
        clue: str,
        guess: Sequence[str],
        not_guess: Sequence[str],
    ) -> float:
        # TODO: do gradient step accumulation either here or within Pragmatic Giver
        self.optim.zero_grad()
        clue_embed = self._encode_and_embed([clue]).to(self.device)
        guess_embed = self._encode_and_embed(guess).to(self.device)
        not_guess_embed = self._encode_and_embed(not_guess).to(self.device)

        loss = self.embedding_loss(clue_embed, guess_embed, not_guess_embed)
        loss.backward()
        self.optim.step()

        return loss.item()

    def train_batch(
        self,
        clue_batch: Sequence[str],
        guess_batch: Sequence[str],
        not_guess_batch: Sequence[str],
    ):
        self.optim.zero_grad()
        loss = 0

        for clue, guess, not_guess in zip(clue_batch, guess_batch, not_guess_batch):
            clue_embed = self._encode_and_embed([clue]).to(self.device)
            guess_embed = self._encode_and_embed(guess.split(',')).to(self.device)
            not_guess_embed = self._encode_and_embed(not_guess.split(',')).to(self.device)
            loss += self.embedding_loss(clue_embed, guess_embed, not_guess_embed)

        loss.backward()
        self.optim.step()

        return loss.item() / len(clue_batch)
    
    def load_model(self, path: str):
        self.trainable.load_state_dict(torch.load(path))

class TrainGiverEmbeddings(GloveEmbeddings):

    def __init__(
        self, 
        in_embed_dim: int = 50, 
        out_embed_dim: int = 50,
        loss_fn: str = "cosine_similarity",
        lr: float = 1e-4,
    ):
        super().__init__(embed_dim=in_embed_dim, loss_fn=loss_fn)
        self.glove = GloVe(name='6B', dim=in_embed_dim)
        glove_weights = torch.load(f".vector_cache/glove.6B.{in_embed_dim}d.txt.pt")
        self.embedding = torch.nn.Embedding.from_pretrained(glove_weights[2])

        self.trainable = nn.Sequential(
            nn.Linear(in_embed_dim, out_embed_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainable.to(self.device)
        self.embedding.to(self.device)

        self.optim = torch.optim.Adam(self.trainable.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)

    def load_weights(self, path: str):
        self.trainable = torch.load(path)

    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        """
        with torch.no_grad():
            base_embedding = super()._encode_and_embed(texts)
            base_embedding = base_embedding.to(self.device)
        return self.trainable(base_embedding)

    def train(
        self,
        dataset: GiverEmbeddingDataset,
        batch_size: int = 1,
    ) -> float:
        join_words = dataset.join_words
        losses = []
        if batch_size == 1:
            dataset.join_words = False
            for clue, targets, neutral, avoid in tqdm(dataset):
                loss = self.train_single(clue, targets, neutral, avoid)
                losses.append(loss)
        else:
            dataset.join_words = True
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
            )
            for clue, targets, neutral, avoid in tqdm(dataloader):
                loss = self.train_batch(clue, targets, neutral, avoid) 
                losses.append(loss)
        self.scheduler.step()
        dataset.join_words = join_words

        return np.mean(losses)
    
    def val(
        self,
        dataset: GiverEmbeddingDataset,
    ) -> float:
        losses = []
        for clue, targets, neutral, avoid in tqdm(dataset):
            clue_embed = self.encode_and_embed([clue])
            targets_embed = self.encode_and_embed(targets)
            neutral_embed = self.encode_and_embed(neutral)
            avoid_embed = self.encode_and_embed(avoid)

            loss = self.embedding_loss(clue_embed, targets_embed, neutral_embed, avoid_embed)
            losses.append(loss.item())

        return np.mean(losses)

    def train_single(
        self,
        clue: str,
        targets: Sequence[str],
        neutral: Sequence[str],
        avoid: Sequence[str],
    ) -> float:
        # TODO: do gradient step accumulation either here or within Pragmatic Giver
        self.optim.zero_grad()
        clue_embed = self._encode_and_embed([clue]).to(self.device)
        targets_embed = self._encode_and_embed(targets).to(self.device)
        neutral_embed = self._encode_and_embed(neutral).to(self.device)
        avoid_embed = self._encode_and_embed(avoid).to(self.device)

        loss = self.embedding_loss(clue_embed, targets_embed, neutral_embed, avoid_embed)
        loss.backward()
        self.optim.step()

        return loss.item()

    def train_batch(
        self,
        clue_batch: Sequence[str],
        targets_batch: Sequence[str],
        neutral_batch: Sequence[str],
        avoid_batch: Sequence[str],
    ):
        self.optim.zero_grad()
        loss = 0

        for clue, targets, neutral, avoid in zip(clue_batch, targets_batch, neutral_batch, avoid_batch):
            clue_embed = self._encode_and_embed([clue]).to(self.device)
            targets_embed = self._encode_and_embed(targets.split(',')).to(self.device)
            neutral_embed = self._encode_and_embed(neutral.split(',')).to(self.device)
            avoid_embed = self._encode_and_embed(avoid.split(',')).to(self.device)
            
            loss = self.embedding_loss(clue_embed, targets_embed, neutral_embed, avoid_embed)

        loss.backward()
        self.optim.step()

        return loss.item() / len(clue_batch)
    
    def load_model(self, path: str):
        self.trainable.load_state_dict(torch.load(path))
