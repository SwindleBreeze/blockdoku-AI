# blockdoku/rollout_buffer.py
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, grid_shape_numpy, piece_vector_size, action_size, gae_lambda=0.95, gamma=0.99, device='cpu'):
        self.num_steps = num_steps
        self.grid_shape_numpy = grid_shape_numpy
        self.piece_vector_size = piece_vector_size
        self.action_size = action_size # Not directly stored, but for context
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device

        self.observations_grid = np.zeros((num_steps, *grid_shape_numpy), dtype=np.float32)
        self.observations_pieces = np.zeros((num_steps, piece_vector_size), dtype=np.float32)
        self.actions = np.zeros((num_steps,), dtype=np.int64)
        self.log_probs = np.zeros((num_steps,), dtype=np.float32)
        self.rewards = np.zeros((num_steps,), dtype=np.float32)
        self.dones = np.zeros((num_steps,), dtype=np.float32) # Store as float for (1-dones)
        self.values = np.zeros((num_steps,), dtype=np.float32)
        
        self.advantages = np.zeros((num_steps,), dtype=np.float32)
        self.returns = np.zeros((num_steps,), dtype=np.float32) # For value function target

        self.ptr = 0
        self.path_start_idx = 0
        self.buffer_filled = False


    def add(self, obs_grid, obs_pieces, action, log_prob, reward, done, value):
        if self.ptr >= self.num_steps:
            print("Warning: Rollout buffer overflow. Check num_steps_per_update.")
            return # Or raise error

        self.observations_grid[self.ptr] = obs_grid
        self.observations_pieces[self.ptr] = obs_pieces
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done) # Store as float
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr == self.num_steps:
            self.buffer_filled = True

    def compute_returns_and_advantages(self, last_value_tensor, last_done):
        last_value = last_value_tensor.cpu().numpy().flatten()[0]
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values # Returns for value function training R_t = A_t + V(s_t)

    def get_batch(self, batch_size):
        if not self.buffer_filled:
            # This should not be called if buffer isn't full yet
            # In PPO, we typically fill the buffer entirely then process it
            raise ValueError("Rollout buffer not yet full. Call compute_returns_and_advantages first.")

        # Create indices for shuffling
        indices = np.arange(self.num_steps)
        np.random.shuffle(indices)

        for start_idx in range(0, self.num_steps, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > self.num_steps: # Ensure mini-batch doesn't exceed buffer size
                end_idx = self.num_steps
            
            minibatch_indices = indices[start_idx:end_idx]

            yield (
                torch.tensor(self.observations_grid[minibatch_indices]).permute(0, 3, 1, 2).to(self.device), # NCHW
                torch.tensor(self.observations_pieces[minibatch_indices]).to(self.device),
                torch.tensor(self.actions[minibatch_indices]).to(self.device),
                torch.tensor(self.log_probs[minibatch_indices]).to(self.device),
                torch.tensor(self.advantages[minibatch_indices]).to(self.device),
                torch.tensor(self.returns[minibatch_indices]).to(self.device),
                torch.tensor(self.values[minibatch_indices]).to(self.device) # Old values V(s_t)
            )
    
    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.buffer_filled = False
        # Optionally zero out arrays again if needed, but they'll be overwritten