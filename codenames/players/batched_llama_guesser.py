from typing import Sequence
from codenames.players import BaseGuesser
from llama_exps.llama_wrapper import LlamaWrapper, LLAMA_7B_TEXT_CKPT_DIR, LLAMA_13B_TEXT_CKPT_DIR, LLAMA_TOKENIZER_PATH

class BatchedLlamaGuesser(BaseGuesser):
    def __init__(self, size):
        super().__init__(size)

    def make_guess(
        self, 
        unselected: Sequence[Sequence[str]],
        clue: Sequence[str],
        num_targets: Sequence[int],
    ):
        
        pass