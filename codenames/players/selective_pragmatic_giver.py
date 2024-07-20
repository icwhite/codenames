from codenames.players import PragmaticGiver
from codenames.players.literal_guesser import LiteralGuesser
from codenames.embeddings import *

class SelectivePragmaticGiver(PragmaticGiver):
    def __init__(
        self, 
        literal_guesser: LiteralGuesser, 
        embeddings: TrainGuesserEmbeddings, 
        adaptive: bool = False,
        choose_argmax: bool = False,
        batch_size: int = 64, 
        buffer_size: int = 10000, 
        k:int = 23
    ):
        super().__init__(literal_guesser,
                          embeddings, 
                          adaptive,
                          choose_argmax,
                          batch_size,
                          buffer_size,
                          k)

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
        loss = 0
        if self.adaptive:
            if guess != targets:
                self.replay_buffer.add_transition(clue, ",".join(guess), ",".join(targets))
                self.num_observations += 1
            if self.num_observations == self.batch_size:
                self.num_observations = 0 
                batch_clue, batch_guess, batch_not_guess = self.replay_buffer.sample_batch(self.batch_size)
                loss = self.embeddings.train_batch(batch_clue, batch_guess, batch_not_guess)
        return {
            "train_loss": loss,
        }