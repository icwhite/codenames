import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.clue_buffer = []
        self.guess_buffer = []
        self.not_guess_buffer = []

    def add_transition(self, clue, guess, not_guess):
        if len(self.clue_buffer) >= self.buffer_size:
            self.clue_buffer.pop(0)
            self.guess_buffer.pop(0)
            self.not_guess_buffer.pop(0)

        self.clue_buffer.append(clue)
        self.guess_buffer.append(guess)
        self.not_guess_buffer.append(not_guess)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.clue_buffer), size=batch_size, replace=False)
        batch_clue = [self.clue_buffer[i] for i in indices]
        batch_guess = [self.guess_buffer[i] for i in indices]
        batch_not_guess = [self.not_guess_buffer[i] for i in indices]
        return batch_clue, batch_guess, batch_not_guess

    def __len__(self):
        return len(self.clue_buffer)
    
if __name__ == "__main__":
    # Example usage:
    buffer_size = 1000
    replay_buffer = ReplayBuffer(buffer_size)

    # Add transitions
    for i in range(100):
        replay_buffer.add_transition(f"clue{i}", f"guess{i}", f"not_guess{i}")

    # Sample a batch
    batch_clue, batch_guess, batch_not_guess = replay_buffer.sample_batch(32)
    print(batch_clue, batch_guess, batch_not_guess)