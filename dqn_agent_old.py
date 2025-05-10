# dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F # Not strictly needed for SmoothL1Loss
import random
import os
# from collections import deque # Not used directly in this file if ReplayBuffer is separate
from torch.optim.lr_scheduler import MultiStepLR

import settings as s # Root AI settings
from replay_buffer import StandardReplayBuffer,PrioritizedReplayBuffer # Assuming this is correctly implemented
# utils.py is not directly imported here usually; its functions are used by env or train script
import torch.nn.functional as F # For activation functions and other operations

# --- Define QNetwork for composite state ---
class QNetwork(nn.Module):
    def __init__(self, grid_shape_pytorch, piece_vector_size, action_size):
        super(QNetwork, self).__init__()
        grid_channels = grid_shape_pytorch[0]
        
        # Enhanced CNN with residual connection
        self.conv1 = nn.Conv2d(grid_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Piece processing pathway
        self.piece_fc = nn.Linear(piece_vector_size, 64)
        self.piece_bn = nn.BatchNorm1d(64)
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_grid = torch.zeros(1, *grid_shape_pytorch)
            x = F.relu(self.bn1(self.conv1(dummy_grid)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            cnn_out_size = x.flatten(1).shape[1]
        
        # Combined pathway
        self.fc1 = nn.Linear(cnn_out_size + 64, 256)
        self.fc_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, grid_input, pieces_input):
        # Grid pathway with residual connection
        x = F.relu(self.bn1(self.conv1(grid_input)))
        residual = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + residual  # Residual connection
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        
        # Piece pathway
        p = F.relu(self.piece_bn(self.piece_fc(pieces_input)))
        
        # Combined pathways
        combined = torch.cat((x, p), dim=1)
        combined = F.relu(self.fc_bn(self.fc1(combined)))
        combined = self.dropout(combined)
        combined = F.relu(self.fc2(combined))
        combined = self.dropout(combined)
        
        return self.fc3(combined)
    
class DQNAgent:
    def __init__(self, grid_observation_shape, piece_vector_size, action_size, load_model_path=None):
        self.grid_shape_numpy = grid_observation_shape # (H, W, C)
        self.grid_shape_pytorch = (grid_observation_shape[-1], grid_observation_shape[0], grid_observation_shape[1]) # (C, H, W)
        self.piece_vector_size = piece_vector_size
        self.action_size = action_size

        self.gamma = s.GAMMA
        self.epsilon = s.EPSILON_START
        self.epsilon_min = s.EPSILON_END
        self.epsilon_decay_steps = s.EPSILON_DECAY_STEPS
        self.learning_rate = s.LEARNING_RATE
        # self.buffer = ReplayBuffer(s.REPLAY_BUFFER_SIZE)
        self.batch_size = s.BATCH_SIZE
        self.target_update_freq = s.TARGET_UPDATE_FREQ
        self.learn_starts = s.LEARNING_STARTS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent using device: {self.device}")

                # --- Initialize correct buffer based on settings ---
        if s.PER_ENABLED:
            print("Using Prioritized Experience Replay (PER)")
            self.buffer = PrioritizedReplayBuffer(
                capacity=s.REPLAY_BUFFER_SIZE,
                alpha=s.PER_ALPHA,
                beta_start=s.PER_BETA_START,
                beta_end=s.PER_BETA_END,
                beta_annealing_steps=s.PER_BETA_ANNEALING_STEPS,
                epsilon=s.PER_EPSILON
            )
        else:
            print("Using Standard Experience Replay")
            self.buffer = StandardReplayBuffer(capacity=s.REPLAY_BUFFER_SIZE)
        # ---------------------------------------------------


        self.model = QNetwork(self.grid_shape_pytorch, self.piece_vector_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.grid_shape_pytorch, self.piece_vector_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=s.LEARNING_RATE, weight_decay=s.WEIGHT_DECAY)
        self.scheduler = MultiStepLR(
            self.optimizer,
            # Strategic milestones based on your training curve
            milestones=s.SCH_MILESTONE,
            gamma=s.SCH_GAMA,  # Reduce LR by 50% at each milestone
            verbose=True  # Print when LR changes
        )

        # self.criterion = nn.SmoothL1Loss() # Huber Loss

        # if load_model_path and os.path.exists(load_model_path):
        #     print(f"Loading model from {load_model_path}")
        #     self.load(load_model_path)
        #     self.update_target_network()
        #     # Optionally set epsilon lower if fine-tuning a well-trained model
        #     # self.epsilon = self.epsilon_min 
        # else:
        #     print("Initializing new model.")
        #     self.update_target_network()

        # self.target_model.eval()
        # self.total_steps = 0

        # Loss function: Use reduction='none' for PER to apply weights
        # If not using PER, can use default reduction='mean'
        reduction_type = 'none' if s.PER_ENABLED else 'mean'
        self.criterion = nn.SmoothL1Loss(reduction=reduction_type) 

        # ... (model loading, target network update, total_steps init) ...
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading model from {load_model_path}")
            self.load(load_model_path)
        else:
            print("Initializing new model.")
        self.update_target_network() # Initialize / Sync target model
        self.target_model.eval()
        self.total_steps = 0


    def remember(self, state_dict, action, reward, next_state_dict, done):
        self.buffer.add(state_dict, action, reward, next_state_dict, done)

    def act(self, state_dict, valid_action_mask=None, use_epsilon=True):
        self.total_steps += 1 # Increment total_steps for epsilon decay
        self._update_epsilon()

        if use_epsilon and np.random.rand() <= self.epsilon:
            if valid_action_mask is not None:
                valid_indices = np.where(valid_action_mask)[0]
                if len(valid_indices) > 0:
                    return random.choice(valid_indices)
                else:
                    # No valid moves according to mask, this case should be handled by game ending
                    # but return a default action if forced to choose.
                    return 0 
            else: # Should not happen if env always provides mask
                return random.randrange(self.action_size)
        
        grid_numpy = state_dict["grid"]
        pieces_numpy = state_dict["pieces"]
        
        # Correct preprocessing for 'act' method (permute and add batch dim)
        grid_tensor = torch.from_numpy(grid_numpy).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        pieces_tensor = torch.from_numpy(pieces_numpy).float().unsqueeze(0).to(self.device)
        
        self.model.eval() # Set model to evaluation mode for inference
        with torch.no_grad():
            act_values = self.model(grid_tensor, pieces_tensor)
        self.model.train() # Set model back to training mode

        act_values_np = act_values.cpu().numpy()[0]

        if valid_action_mask is not None:
            if len(valid_action_mask) != self.action_size:
                 print(f"Warning: Mask size ({len(valid_action_mask)}) != Action size ({self.action_size}) in act()")
                 return int(np.argmax(act_values_np)) # Fallback if mask is wrong size

            masked_act_values = np.where(valid_action_mask, act_values_np, -np.inf)
            best_action = np.argmax(masked_act_values)
            
            if np.isneginf(masked_act_values[best_action]):
                # If all valid actions have -inf Q-value, or no valid actions (shouldn't happen if game logic is right)
                # Fallback to a random valid action if any exist
                valid_indices = np.where(valid_action_mask)[0]
                if len(valid_indices) > 0:
                    # print("Warning: All valid actions have -inf Q-value, picking random valid action.")
                    return random.choice(valid_indices)
                else:
                    # print("Warning: No valid actions available in act method, returning 0.")
                    return 0 # Should ideally be caught by game over
            return int(best_action)
        else: # Should ideally always have a mask from the environment
             return int(np.argmax(act_values_np))

    def replay(self):
        # --- Modified for PER ---
        if len(self.buffer) < self.learn_starts or len(self.buffer) < self.batch_size:
            return 0.0 

        # Sample batch - PER buffer returns experiences, indices, and IS weights
        # Standard buffer returns experiences, dummy indices, and weights=1.0
        experiences_tuple, indices, is_weights = self.buffer.sample(self.batch_size)
        states_dict_tuple, actions_tuple, rewards_tuple, next_states_dict_tuple, dones_tuple = experiences_tuple

        # --- Batch Data Processing (remains similar) ---
        grids_np = np.array([s['grid'] for s in states_dict_tuple])            
        pieces_np = np.array([s['pieces'] for s in states_dict_tuple])         
        next_grids_np = np.array([s['grid'] for s in next_states_dict_tuple])  
        next_pieces_np = np.array([s['pieces'] for s in next_states_dict_tuple])

        actions_np = np.array(actions_tuple)
        rewards_np = np.array(rewards_tuple)
        dones_np = np.array(dones_tuple)

        grids = torch.from_numpy(grids_np).float().permute(0, 3, 1, 2).to(self.device) 
        pieces = torch.from_numpy(pieces_np).float().to(self.device) 
        actions = torch.from_numpy(actions_np).long().unsqueeze(1).to(self.device) 
        rewards = torch.from_numpy(rewards_np).float().unsqueeze(1).to(self.device) 
        next_grids = torch.from_numpy(next_grids_np).float().permute(0, 3, 1, 2).to(self.device)
        next_pieces = torch.from_numpy(next_pieces_np).float().to(self.device) 
        dones = torch.from_numpy(dones_np).float().unsqueeze(1).to(self.device) 
        # --- PER: Convert IS weights to tensor ---
        is_weights_tensor = torch.from_numpy(is_weights).float().unsqueeze(1).to(self.device)
        
        # --- Target Q-value calculation (Double DQN - remains the same) ---
        self.model.eval() 
        self.target_model.eval()
        with torch.no_grad():
            next_q_values_main_model = self.model(next_grids, next_pieces)
            best_next_actions = next_q_values_main_model.argmax(dim=1, keepdim=True)
            
            next_q_values_target_model = self.target_model(next_grids, next_pieces)
            target_q_subset = next_q_values_target_model.gather(1, best_next_actions)
            
            target_q_values = rewards + self.gamma * target_q_subset * (1 - dones)

        # --- Current Q-value prediction ---
        self.model.train() 
        q_values = self.model(grids, pieces)
        action_q_values = q_values.gather(1, actions)

        
        # --- DEBUG PRINTS (Uncomment to diagnose extreme loss values) ---
        # print(f"--- Replay Batch Debug ---")
        # print(f"Rewards: min={rewards.min().item():.3f}, max={rewards.max().item():.3f}, mean={rewards.mean().item():.3f}")
        # print(f"Dones sum: {dones.sum().item()}")
        # print(f"TargetNet Q_subset: min={target_q_subset.min().item():.3f}, max={target_q_subset.max().item():.3f}, mean={target_q_subset.mean().item():.3f}")
        # print(f"TD Targets: min={target_q_values.min().item():.3f}, max={target_q_values.max().item():.3f}, mean={target_q_values.mean().item():.3f}")
        # print(f"Predicted Qs: min={action_q_values.min().item():.3f}, max={action_q_values.max().item():.3f}, mean={action_q_values.mean().item():.3f}")
        # --- End Debug Prints ---

        # --- Calculate Loss (Modified for PER) ---
        # Calculate element-wise loss (since reduction='none')
        elementwise_loss = self.criterion(action_q_values, target_q_values)
        
        # Apply Importance Sampling weights
        loss = (is_weights_tensor * elementwise_loss).mean() # Weighted mean

        # --- Calculate TD errors to update priorities ---
        # Use detach() to prevent gradients flowing back from this calculation
        td_errors_abs = torch.abs(target_q_values - action_q_values).detach().cpu().numpy().flatten()

        # --- Optimize the Model ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=s.GRADIENT_CLIP_NORM)
        self.optimizer.step()

        # --- Update Priorities in PER Buffer ---
        if s.PER_ENABLED:
             self.buffer.update_priorities(indices, td_errors_abs) # Pass indices and errors

        # --- Update Target Network ---
        if self.total_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _update_epsilon(self):
        # Calculate steps since learning started
        effective_decay_steps = max(0, self.total_steps - self.learn_starts)
        
        if effective_decay_steps < self.epsilon_decay_steps:
            # Use polynomial decay for smoother transition
            # Power determines how gradual the decay is (higher = more gradual)
            
            progress = effective_decay_steps / self.epsilon_decay_steps
            self.epsilon = s.EPSILON_END + (s.EPSILON_START - s.EPSILON_END) * ((1 - progress) ** s.POWER_DECAY)
        else:
            # Add a small periodic bump to epsilon to encourage occasional exploration
            # even after the main decay period
            base_epsilon = s.EPSILON_END
            bump = s.BUMP_SIZE * (np.sin(self.total_steps / s.BUMP_PERIOD) * 0.5 + 0.5)
            self.epsilon = base_epsilon + bump
        
        # Ensure epsilon stays within bounds
        self.epsilon = max(self.epsilon_min, min(s.EPSILON_START, self.epsilon))


    def load(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            # self.target_model.load_state_dict(self.model.state_dict()) # Also update target after loading
            print(f"Model weights loaded from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        # print(f"Model weights saved to {path}") # Handled in train.py
    # Add a step method to your agent
    
    def scheduler_step(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            return self.optimizer.param_groups[0]['lr']
        return None