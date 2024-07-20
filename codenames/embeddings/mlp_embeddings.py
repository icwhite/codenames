from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from .embedding_dataset import GuesserEmbeddingDataset
from .word_embeddings import Embeddings
from IPython import embed

class MLPEmbeddings(Embeddings):

    def __init__(
        self, 
        arch: Sequence[int] = [1, 50, 100, 500],
        loss_fn: str = "cosine_similarity",
        lr: float = 1e-4,
    ):
        super().__init__(loss_fn=loss_fn)

        layers = [nn.Linear(arch[0], arch[1])]
        for i in range(1, len(arch) - 1):
            layers.extend([
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(arch[i], arch[i+1]),
            ])
        self.model = nn.Sequential(*layers)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)

    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        """
        """
        texts: list of strings, the texts to embed
        out: size (len(texts), 1, embedding_dim) or (num_target_pairs, num_targets+tokens, embedding_dim)
        """

        encoding = self.tokenizer.batch_encode_plus(
            texts,                     # List of input texts
            padding=True,              # Pad to the maximum sequence length
            truncation=True,           # Truncate to the maximum sequence length if necessary
            return_tensors='pt',       # Return PyTorch tensors
            add_special_tokens=False   # Don't add special tokens CLS and SEP
        )
        
        input_ids = encoding['input_ids'].float().to(self.device)  # Token IDs
        input_ids = torch.mean(input_ids, axis=1, keepdim=True)

        embeddings = self.model(input_ids)
        return torch.unsqueeze(embeddings, dim=1)

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