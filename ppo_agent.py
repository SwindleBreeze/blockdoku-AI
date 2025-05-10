# blockdoku/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import os

import settings as s_ai # Root AI settings
from rollout_buffer import RolloutBuffer 

## ORIGINAL CODE - commented out for debugging
# class ActorCritic(nn.Module):
#     def __init__(self, grid_shape_pytorch, piece_vector_size, action_size):
#         super(ActorCritic, self).__init__()
#         grid_channels = grid_shape_pytorch[0]

#         # Shared CNN layers
#         self.shared_cnn = nn.Sequential(
#             nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         with torch.no_grad():
#             dummy_grid = torch.zeros(1, *grid_shape_pytorch)
#             cnn_out_size = self.shared_cnn(dummy_grid).shape[1]

#         # Shared FC layers before actor/critic heads
#         self.shared_fc = nn.Sequential(
#             nn.Linear(cnn_out_size + piece_vector_size, 256), # Increased size
#             nn.ReLU()
#         )
        
#         # Actor head (outputs action logits)
#         self.actor_head = nn.Linear(256, action_size)
#         # Critic head (outputs state value)
#         self.critic_head = nn.Linear(256, 1)

#     def forward(self, grid_input, pieces_input):
#         cnn_features = self.shared_cnn(grid_input)
#         combined_features = torch.cat((cnn_features, pieces_input), dim=1)
#         shared_out = self.shared_fc(combined_features)
        
#         action_logits = self.actor_head(shared_out)
#         value = self.critic_head(shared_out)
        
#         return action_logits, value

# Updated ActorCritic class for better performance
class ActorCritic(nn.Module):
    def __init__(self, grid_shape_pytorch, piece_vector_size, action_size):
        super(ActorCritic, self).__init__()
        grid_channels = grid_shape_pytorch[0]

        # --- Smaller Shared CNN ---
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1), # Fewer channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Fewer channels
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Removed one layer
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_grid = torch.zeros(1, *grid_shape_pytorch)
            cnn_out_size = self.shared_cnn(dummy_grid).shape[1]

        # --- Smaller Shared FC ---
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out_size + piece_vector_size, 64), # Smaller FC layer
            nn.ReLU()
        )
        
        # Actor head (input size matches smaller shared_fc output)
        self.actor_head = nn.Linear(64, action_size)
        # Critic head (input size matches smaller shared_fc output)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, grid_input, pieces_input):
        cnn_features = self.shared_cnn(grid_input)
        combined_features = torch.cat((cnn_features, pieces_input), dim=1)
        shared_out = self.shared_fc(combined_features)
        
        action_logits = self.actor_head(shared_out)
        value = self.critic_head(shared_out)
        
        return action_logits, value

class PPOAgent:
    def __init__(self, grid_observation_shape, piece_vector_size, action_size, load_model_path=None):
        self.grid_shape_numpy = grid_observation_shape
        self.grid_shape_pytorch = (grid_observation_shape[-1], grid_observation_shape[0], grid_observation_shape[1])
        self.piece_vector_size = piece_vector_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.actor_critic = ActorCritic(self.grid_shape_pytorch, self.piece_vector_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=s_ai.PPO_LEARNING_RATE, eps=1e-5) # Added eps for Adam

        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading PPO model from {load_model_path}")
            self.load_model(load_model_path)
        else:
            print("Initializing new PPO model.")

    def get_action_and_value(self, state_dict, action_mask=None, action_to_take=None):
        grid_numpy = state_dict["grid"]
        pieces_numpy = state_dict["pieces"]
        
        grid_tensor = torch.from_numpy(grid_numpy).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        pieces_tensor = torch.from_numpy(pieces_numpy).float().unsqueeze(0).to(self.device)
        
        logits, value = self.actor_critic(grid_tensor, pieces_tensor) # Get logits and value

        if action_mask is not None:
            # Apply mask to logits before creating distribution
            # Convert mask to tensor and ensure it's on the same device
            mask_tensor = torch.from_numpy(action_mask).bool().to(self.device)
            if mask_tensor.shape[0] != logits.shape[1]:
                 print(f"Warning: Mask shape {mask_tensor.shape} doesn't match logits shape {logits.shape} in get_action_and_value. Using unmasked logits.")
            else:
                logits[0, ~mask_tensor] = -float('inf') # Set logits of invalid actions to -inf
        
        probs = Categorical(logits=logits) # Create distribution from (potentially masked) logits
        
        if action_to_take is None:
            action = probs.sample() # Sample an action
        else:
            action = torch.tensor([action_to_take]).to(self.device) # Use provided action (for update phase)

        log_prob = probs.log_prob(action)
        
        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()

    def get_value(self, state_dict):
        grid_numpy = state_dict["grid"]
        pieces_numpy = state_dict["pieces"]
        grid_tensor = torch.from_numpy(grid_numpy).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        pieces_tensor = torch.from_numpy(pieces_numpy).float().unsqueeze(0).to(self.device)
        
        _, value = self.actor_critic(grid_tensor, pieces_tensor)
        return value # Return as tensor

    def update(self, rollout_buffer: RolloutBuffer):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        # Normalize advantages (optional but often helpful)
        advantages = rollout_buffer.advantages # Get advantages calculated by the buffer
        # Convert to tensor first, then normalize
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        # Now advantages_tensor is what we will use for policy loss


        for _ in range(s_ai.PPO_EPOCHS):
            for batch_obs_grid, batch_obs_pieces, batch_actions, batch_old_log_probs, \
                batch_advantages_from_buffer, batch_returns, batch_old_values \
                in rollout_buffer.get_batch(s_ai.PPO_MINI_BATCH_SIZE): # This yields tensors already on device

                # We need the corresponding advantages for this minibatch.
                # The get_batch should ideally yield the pre-normalized advantages for the minibatch.
                # Let's assume get_batch yields slices of the *normalized* advantages_tensor we created above.
                # To do this correctly, RolloutBuffer.get_batch needs access to the normalized advantages.
                # For simplicity here, let's assume the batch_advantages from buffer *is* the one to use,
                # and it should have been normalized if we decide to normalize.
                # The code for `get_batch` currently yields self.advantages[minibatch_indices] directly.
                # So, the normalization should be done on self.advantages in the buffer, OR we pass
                # the normalized advantages tensor to get_batch or index it here.

                # Correct approach: Normalize all advantages first, then get_batch yields slices of this.
                # So, in get_batch, it should yield `advantages_tensor[minibatch_indices]`
                # Let's modify RolloutBuffer.get_batch slightly to accept the normalized advantages.
                # OR, simpler for now: use batch_advantages_from_buffer and ensure it's normalized if desired.
                # The current PPO_Agent.update takes the raw advantages from buffer and normalizes them.
                # So, we need to ensure this normalized version is used for each minibatch.
                # For now, I'll use batch_advantages_from_buffer directly, assuming it's already what we want (or normalized prior if strategy changes)
                
                # Let's use the already sliced batch_advantages_from_buffer assuming it's correct for this minibatch
                # and if normalization is applied, it happens to the whole buffer before slicing.
                # For this example, `batch_advantages_from_buffer` *is* already a tensor.
                # My previous normalization created `advantages_tensor` for the *whole* rollout.
                # The mini-batching in `get_batch` already slices `self.advantages`.
                # So, the normalization must happen *before* mini-batching or on the mini-batch itself.
                # Let's normalize the `batch_advantages_from_buffer` for this specific mini-batch:
                
                batch_advantages_normalized = batch_advantages_from_buffer


                # Reshape actions if necessary, they should be (N,) for log_prob
                batch_actions = batch_actions.squeeze() 
                
                new_logits, new_values = self.actor_critic(batch_obs_grid, batch_obs_pieces)
                new_probs = Categorical(logits=new_logits)
                new_log_probs = new_probs.log_prob(batch_actions)
                entropy = new_probs.entropy().mean()

                # Policy Loss (Clipped Surrogate Objective)
                logratio = new_log_probs - batch_old_log_probs # batch_old_log_probs from buffer
                ratio = torch.exp(logratio)
                
                # Use the normalized advantages for this batch
                pg_loss1 = batch_advantages_normalized * ratio
                pg_loss2 = batch_advantages_normalized * torch.clamp(ratio, 1 - s_ai.PPO_CLIP_EPSILON, 1 + s_ai.PPO_CLIP_EPSILON)
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                # Value Loss
                new_values = new_values.view(-1) # Ensure new_values is (N,)
                # Clipped value loss (optional, but common in PPO implementations)
                value_pred_clipped = batch_old_values + \
                    torch.clamp(new_values - batch_old_values, -s_ai.PPO_CLIP_EPSILON, s_ai.PPO_CLIP_EPSILON)
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                # Simpler value loss (if not using clipped value loss):
                # value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()


                # Total Loss
                loss = policy_loss - s_ai.PPO_ENTROPY_COEF * entropy + s_ai.PPO_VALUE_LOSS_COEF * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), s_ai.PPO_MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
        
        num_updates_in_epoch = s_ai.PPO_EPOCHS * (rollout_buffer.num_steps // s_ai.PPO_MINI_BATCH_SIZE)
        if num_updates_in_epoch == 0: num_updates_in_epoch = 1 # Avoid division by zero if buffer < minibatch

        avg_policy_loss = total_policy_loss / num_updates_in_epoch
        avg_value_loss = total_value_loss / num_updates_in_epoch
        avg_entropy_loss = total_entropy_loss / num_updates_in_epoch
        
        return avg_policy_loss, avg_value_loss, avg_entropy_loss


    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor_critic.state_dict(), path)
        print(f"PPO Model saved to {path}")

    def load_model(self, path):
        try:
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
            print(f"PPO Model loaded from {path}")
        except Exception as e:
            print(f"Error loading PPO model: {e}")