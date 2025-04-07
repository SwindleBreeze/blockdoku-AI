# dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # For loss functions if needed, or use nn.MSELoss
import random
import os
from collections import deque # Re-import deque if not using separate file

import settings as s
# Assuming replay_buffer.py exists and works with numpy arrays
from replay_buffer import ReplayBuffer
from utils import get_valid_action_mask

# Define the Q-Network architecture using PyTorch
# --- Define QNetwork for composite state ---
class QNetwork(nn.Module):
    def __init__(self, grid_shape_pytorch, piece_vector_size, action_size):
        super(QNetwork, self).__init__()
        # Grid processing branch (CNN)
        grid_channels = grid_shape_pytorch[0]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened CNN size
        with torch.no_grad():
            dummy_grid = torch.zeros(1, *grid_shape_pytorch)
            cnn_out_size = self.conv_layers(dummy_grid).shape[1]

        # Combined processing branch (Dense)
        self.fc_layers = nn.Sequential(
            # Input size = flattened CNN output + piece vector size
            nn.Linear(cnn_out_size + piece_vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, grid_input, pieces_input):
        # Process grid through CNN
        cnn_features = self.conv_layers(grid_input)
        # Concatenate CNN features and piece vector features
        combined_features = torch.cat((cnn_features, pieces_input), dim=1)
        # Process combined features through Dense layers
        q_values = self.fc_layers(combined_features)
        return q_values

class DQNAgent:
    def __init__(self, grid_observation_shape, piece_vector_size, action_size, load_model_path=None):
        # Store shapes separately
        self.grid_shape_numpy = grid_observation_shape # (H, W, C)
        self.grid_shape_pytorch = (grid_observation_shape[-1], grid_observation_shape[0], grid_observation_shape[1]) # (C, H, W)
        self.piece_vector_size = piece_vector_size
        self.action_size = action_size

        # Hyperparameters from settings
        self.gamma = s.GAMMA
        self.epsilon = s.EPSILON_START
        self.epsilon_min = s.EPSILON_END
        self.epsilon_decay_steps = s.EPSILON_DECAY_STEPS
        self.learning_rate = s.LEARNING_RATE
        self.buffer = ReplayBuffer(s.REPLAY_BUFFER_SIZE)
        self.batch_size = s.BATCH_SIZE
        self.target_update_freq = s.TARGET_UPDATE_FREQ
        self.learn_starts = s.LEARNING_STARTS

        # Setup device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Build networks with new architecture
        self.model = QNetwork(self.grid_shape_pytorch, self.piece_vector_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.grid_shape_pytorch, self.piece_vector_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=s.LEARNING_RATE)
        self.criterion = nn.MSELoss()

        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading model from {load_model_path}")
            self.load(load_model_path) # Load state dict
            self.update_target_network() # Sync target model
            self.epsilon = self.epsilon_min # Start with low epsilon
        else:
            print("Initializing new model.")
            self.update_target_network() # Initialize target model weights

        self.target_model.eval() # Target network is only for inference

        self.total_steps = 0

    def _preprocess_state(self, state_numpy):
        """ Converts state from env (H, W, C) to PyTorch tensor (1, C, H, W)."""
        # Add batch dim, permute, convert to tensor, move to device
        state_tensor = torch.from_numpy(state_numpy).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return state_tensor

    def remember(self, state_dict, action, reward, next_state_dict, done):
        # Store the state dictionary directly
        self.buffer.add(state_dict, action, reward, next_state_dict, done)

    def act(self, state_dict, valid_action_mask=None, use_epsilon=True):
        """Chooses an action using epsilon-greedy policy."""
        self.total_steps += 1
        self._update_epsilon()

        # ! CHECK IF makes sense
        if use_epsilon and np.random.rand() <= self.epsilon:
            # Explore: Choose a random *valid* action
            if valid_action_mask is not None:
                valid_indices = np.where(valid_action_mask)[0]
                if len(valid_indices) > 0:
                    return random.choice(valid_indices) #? Choose a random valid action
                else:
                    return 0 # No valid moves
            else:
                return random.randrange(self.action_size) # Fallback #!

        # Exploit: Choose the best *valid* action based on Q-values
        # Exploitation
        # Prepare state tensors from dict
        grid_numpy = state_dict["grid"] # (H, W, C)
        pieces_numpy = state_dict["pieces"] # (P,) P = num_piece_types

        grid_tensor = torch.from_numpy(grid_numpy).float().permute(2, 0, 1).unsqueeze(0).to(self.device) # (1, C, H, W)
        pieces_tensor = torch.from_numpy(pieces_numpy).float().unsqueeze(0).to(self.device) # (1, P)

        # ??? 
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(grid_tensor, pieces_tensor) # Pass both inputs
        self.model.train()

        act_values_np = act_values.cpu().numpy()[0] # Get numpy array on CPU

        # Apply mask and choose best valid action
        if valid_action_mask is not None:
            if len(valid_action_mask) != self.action_size:
                 print(f"Warning: Mask size ({len(valid_action_mask)}) != Action size ({self.action_size})")
                 best_action = np.argmax(act_values_np) # Proceed without mask
            else:
                 act_values_np[~valid_action_mask] = -np.inf
                 best_action = np.argmax(act_values_np)
                 if np.isneginf(act_values_np[best_action]):
                     return 0 # No valid moves
                 return int(best_action) # Return as int
        else:
             return int(np.argmax(act_values_np)) # Return as int

    def replay(self):
        if len(self.buffer) < self.batch_size or self.total_steps < self.learn_starts:
            return 0.0

        # Sample batch (returns list of tuples)
        experiences = self.buffer.sample(self.batch_size)

        # --- Unpack experiences and process Batch Data ---
        # Use zip(*experiences) to transpose the list of tuples
        states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple = zip(*experiences)

        # Extract grid and pieces from dicts and convert to NumPy arrays
        grids_np = np.array([s['grid'] for s in states_dict_tuple])             # Shape: (N, H, W, C)
        pieces_np = np.array([s['pieces'] for s in states_dict_tuple])          # Shape: (N, P)
        next_grids_np = np.array([s['grid'] for s in next_states_dict_tuple])   # Shape: (N, H, W, C)
        next_pieces_np = np.array([s['pieces'] for s in next_states_dict_tuple]) # Shape: (N, P)

        # Convert actions, rewards, dones to NumPy arrays
        actions_np = np.array(actions_tuple)
        rewards_np = np.array(rewards_tuple)
        dones_np = np.array(dones_tuple)

        # --- Convert NumPy arrays to PyTorch tensors (as before) ---
        grids = torch.from_numpy(grids_np).float().permute(0, 3, 1, 2).to(self.device) # (N, C, H, W)
        pieces = torch.from_numpy(pieces_np).float().to(self.device) # (N, P)
        actions = torch.from_numpy(actions_np).long().unsqueeze(1).to(self.device) # (N, 1) Long
        rewards = torch.from_numpy(rewards_np).float().unsqueeze(1).to(self.device) # (N, 1)
        next_grids = torch.from_numpy(next_grids_np).float().permute(0, 3, 1, 2).to(self.device) # (N, C, H, W)
        next_pieces = torch.from_numpy(next_pieces_np).float().to(self.device) # (N, P)
        dones = torch.from_numpy(dones_np).float().unsqueeze(1).to(self.device) # (N, 1) Float

        # --- Calculate Target Q-values (Double DQN) ---
        # (Rest of the target calculation code remains the same)
        with torch.no_grad():
            next_q_values_main = self.model(next_grids, next_pieces)
            best_next_actions = next_q_values_main.argmax(dim=1, keepdim=True)
            next_q_values_target = self.target_model(next_grids, next_pieces)
            target_q_subset = next_q_values_target.gather(1, best_next_actions)
            target_q_values = rewards + self.gamma * target_q_subset * (1 - dones)

        # --- Calculate Current Q-values ---
        # (Remains the same)
        q_values = self.model(grids, pieces)
        action_q_values = q_values.gather(1, actions)

        # --- Calculate Loss ---
        # (Remains the same)
        loss = self.criterion(action_q_values, target_q_values)

        # --- Optimize the Model ---
        # (Remains the same: zero_grad, backward, clip_grad, step)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=s.GRADIENT_CLIP_NORM)
        self.optimizer.step()

        # --- Update Target Network ---
        # (Remains the same)
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Copies weights from the main model to the target model."""
        #print("Updating target network...")
        self.target_model.load_state_dict(self.model.state_dict())

    def _update_epsilon(self):
        """Decays epsilon linearly."""
        if self.total_steps < self.epsilon_decay_steps:
            self.epsilon = s.EPSILON_START - (s.EPSILON_START - s.EPSILON_END) * (self.total_steps / self.epsilon_decay_steps)
        else:
            self.epsilon = s.EPSILON_END
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def load(self, path):
        try:
            # Load state dict, ensuring it's mapped to the correct device
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model weights loaded from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save only the state dict
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")