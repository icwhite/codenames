from codenames.players.pragmatic_giver import PragmaticGiver
from codenames.embeddings import TrainGuesserEmbeddings
from codenames.embeddings.mixture_embeddings import MixtureEmbeddings
from typing import List, Sequence

class CrossCulturalGiver(PragmaticGiver):

    def __init__(
        self, 
        literal_guesser, 
        embeddings, 
        alpha=0.7,
    ):
        super().__init__(literal_guesser, embeddings)
        self.prob_same_culture = 1
        self.alpha = alpha

        self.original_embeddings = embeddings
        self.new_embeddings = TrainGuesserEmbeddings(in_embed_dim=embeddings.in_embed_dim, 
                                              out_embed_dim=embeddings.out_embed_dim,
                                              loss_fn="cosine_similarity",
                                              lr=1e-3)
        self.prob_same_culture = 1
        self.embeddings = MixtureEmbeddings(embeddings, self.new_embeddings, mixture=0.5)
    
    def compute_prob_same_culture(
        self, 
        goal: Sequence[str],
        avoid: Sequence[str],
        neutral: Sequence[str],
        targets: Sequence[str],
        clue: str,
        guess: Sequence[str],
    ):
        unselected = list(set(goal) | set(avoid) | set(neutral))
        total_prob = 1
        for g in guess:
            if g in targets: 
                continue
            guess_probs = self.literal_guesser.guess_probabilities(unselected, clue)
            g_idx = unselected.index(g)
            total_prob *= guess_probs[g_idx]
            unselected.remove(g)
        assert total_prob > 0, total_prob < 1
        return total_prob

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
        prob = self.compute_prob_same_culture(goal, avoid, neutral, targets, clue, guess)
        self.prob_same_culture = self.prob_same_culture * self.alpha + (1 - self.alpha) * prob
        self.embeddings.update_mixture(self.prob_same_culture)
        #TODO: how can we set the learning rate of embeddings to be higher? 
        self.original_embeddings.train_single(clue, guess, not_guess)