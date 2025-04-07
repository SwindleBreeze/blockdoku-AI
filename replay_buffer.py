# replay_buffer.py
import random
from collections import deque
# No numpy import needed here anymore for conversion

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """ Adds an experience to the buffer. state/next_state can be dicts. """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """ Samples a batch of experiences from the buffer. """
        # Simply return the list of sampled experience tuples
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)