from typing import Sequence

import torch
import transformers

from .word_embeddings import Embeddings

class BertEmbeddings(Embeddings):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        model: torch.nn.Module,
        loss_fn: str = "cosine_similarity",
    ):
        # TODO: update to load from checkpoints
        super().__init__(loss_fn=loss_fn)
        self.tokenizer = tokenizer
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.9)
    
    def _encode_and_embed(self, texts: Sequence[str]) -> torch.Tensor:
        encoding = self.tokenizer.batch_encode_plus(
            texts,                     # List of input texts
            padding=True,              # Pad to the maximum sequence length
            truncation=True,           # Truncate to the maximum sequence length if necessary
            return_tensors='pt',       # Return PyTorch tensors
            add_special_tokens=False   # Don't add special tokens CLS and SEP
        )
        
        input_ids = encoding['input_ids']  # Token IDs
        attention_mask = encoding['attention_mask']
        
        outputs = self.model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
        return word_embeddings