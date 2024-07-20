from typing import Sequence

from codenames.players import BaseGuesser
from llama_exps.llama_wrapper import LlamaWrapper, LLAMA_7B_TEXT_CKPT_DIR, LLAMA_13B_TEXT_CKPT_DIR, LLAMA_TOKENIZER_PATH

class LlamaGuesser(BaseGuesser):
    def __init__(self, size):
        super().__init__()
        assert size in ['7b', '13b']
        ckpt_dir = LLAMA_7B_TEXT_CKPT_DIR if size == '7b' else LLAMA_13B_TEXT_CKPT_DIR
        self.generator = LlamaWrapper.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=LLAMA_TOKENIZER_PATH,
            max_seq_len=512,
            max_batch_size=8,
        )

    def make_guess(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
    ) -> Sequence[str]:
        joined_unselected = ", ".join(unselected)
        prompt = f"You are playing Codenames and are the clue guesser. The words on the board are {joined_unselected}. You are given the clue {clue}. The word you would guess is "

        probs = self.generator.get_fixed_output_prob(
            prompts=[prompt],
            responses=unselected,
        )
        zipped = [item for item in zip(unselected, probs)]
        zipped.sort(key = lambda x: x[1], reverse=True)
        assert zipped[0][1] >= zipped[1][1]
        return [i[0] for i in zipped[:num_targets]]

    def observe_turn(
        self,
        unselected: Sequence[str],
        clue: str,
        num_targets: int,
        guess: Sequence[str],
        result: Sequence[str],
    ):
        pass

    def guess_probabilities(self, 
                            unselected: Sequence[str],
                            clue: str) -> Sequence[float]:
        joined_unselected = ", ".join(unselected)
        prompt = f"You are playing Codenames and are the clue guesser. The words on the board are {joined_unselected}. You are given the clue {clue}. The word you would guess is "

        probs = self.generator.get_fixed_output_prob(
            prompts=[prompt],
            responses=unselected,
        )
        return probs
        