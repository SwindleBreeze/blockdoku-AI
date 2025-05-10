# blockdoku/train_ppo.py
import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm 
import torch # For device management if needed here, agent handles its own

import settings as s_ai # Root AI settings
from blockdoku_env import BlockdokuEnv
from ppo_agent import PPOAgent # Import the new PPO agent
from rollout_buffer import RolloutBuffer # Import the new Rollout Buffer
import json
import datetime

def save_ppo_model(agent, filename="BD_PPO", version=None, save_dir=None):
    model_dir = save_dir if save_dir else s_ai.MODEL_SAVE_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = filename
    if version is not None:
        base_name = f"{base_name}_update{version}"
    model_path = os.path.join(model_dir, f"{base_name}.pth")
    agent.save_model(model_path) # PPOAgent has its own save method
    # print(f"PPO Model saved to {model_path}") # Agent prints this
    return model_path

def train_ppo():
    print(f"PPO Training using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}") # Confirm device for main script
    
    if not os.path.exists(s_ai.MODEL_SAVE_DIR):
        os.makedirs(s_ai.MODEL_SAVE_DIR)

    env = BlockdokuEnv(render_mode=None)
    # vis_env = BlockdokuEnv(render_mode="human") # Optional for visualization

    grid_shape_numpy = (s_ai.GRID_HEIGHT, s_ai.GRID_WIDTH, s_ai.STATE_GRID_CHANNELS)
    piece_vector_size = s_ai.STATE_PIECE_VECTOR_SIZE
    
    agent = PPOAgent(grid_shape_numpy, piece_vector_size, env.action_size, load_model_path=None) # Start fresh

    rollout_buffer = RolloutBuffer(
        num_steps=s_ai.PPO_NUM_STEPS_PER_UPDATE,
        grid_shape_numpy=grid_shape_numpy,
        piece_vector_size=piece_vector_size,
        action_size=env.action_size, # Not strictly needed by buffer but good for consistency
        gae_lambda=s_ai.PPO_GAE_LAMBDA,
        gamma=s_ai.PPO_GAMMA,
        device=agent.device 
    )

    episode_rewards_window = deque(maxlen=100)
    game_scores_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_log_path = f"logs/ppo_training_log_{timestamp}.json"
    os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
    
    training_log = {
        "updates_data": [],
        "settings": { # Log PPO specific settings
            "total_timesteps": s_ai.NUM_TOTAL_TIMESTEPS_PPO,
            "learning_rate": s_ai.PPO_LEARNING_RATE,
            "gamma": s_ai.PPO_GAMMA,
            "gae_lambda": s_ai.PPO_GAE_LAMBDA,
            "clip_epsilon": s_ai.PPO_CLIP_EPSILON,
            "epochs": s_ai.PPO_EPOCHS,
            "mini_batch_size": s_ai.PPO_MINI_BATCH_SIZE,
            "value_loss_coef": s_ai.PPO_VALUE_LOSS_COEF,
            "entropy_coef": s_ai.PPO_ENTROPY_COEF,
            "num_steps_per_update": s_ai.PPO_NUM_STEPS_PER_UPDATE,
            # Include relevant reward settings too
            "reward_block_placed": s_ai.REWARD_BLOCK_PLACED,
            "reward_line_square_clear": s_ai.REWARD_LINE_SQUARE_CLEAR,
            "reward_almost_full": s_ai.REWARD_ALMOST_FULL,
            "invalid_move_penalty": s_ai.INVALID_MOVE_PENALTY,
            "stuck_penalty_rl": s_ai.STUCK_PENALTY_RL
        }
    }

    print("Starting PPO Training...")
    
    obs_dict, info = env.reset() # obs_dict is {"grid": grid_np, "pieces": pieces_np}
    current_obs_grid = obs_dict["grid"]
    current_obs_pieces = obs_dict["pieces"]
    current_done = False
    
    num_updates = s_ai.NUM_TOTAL_TIMESTEPS_PPO // s_ai.PPO_NUM_STEPS_PER_UPDATE
    total_timesteps_collected = 0
    
    # For tracking episode metrics
    current_episode_rl_reward = 0
    current_episode_steps = 0
    num_episodes_completed = 0

    pbar = tqdm(range(1, num_updates + 1), initial=1, total=num_updates, unit="update")

    for update_num in pbar:
        rollout_buffer.reset() # Clear buffer for new trajectories

        # --- Collect Trajectories (PPO_NUM_STEPS_PER_UPDATE steps) ---
        for step_in_update in range(s_ai.PPO_NUM_STEPS_PER_UPDATE):
            total_timesteps_collected += 1
            current_episode_steps += 1
            
            # Get action mask
            valid_mask = info.get("valid_action_mask", None)
            if valid_mask is None and hasattr(env, '_get_info'): # Fallback
                valid_mask = env._get_info().get("valid_action_mask")

            # Agent acts (get action, log_prob, value)
            action, log_prob, value = agent.get_action_and_value(
                {"grid": current_obs_grid, "pieces": current_obs_pieces},
                action_mask=valid_mask
            )
            
            next_obs_dict, reward, done, info = env.step(action)
            current_episode_rl_reward += reward
            
            rollout_buffer.add(current_obs_grid, current_obs_pieces, action, log_prob, reward, done, value)
            
            current_obs_grid = next_obs_dict["grid"]
            current_obs_pieces = next_obs_dict["pieces"]
            current_done = done # Store done for GAE calculation if episode ends mid-rollout

            if done:
                episode_rewards_window.append(current_episode_rl_reward)
                game_scores_window.append(info.get("game_score_display", 0))
                steps_window.append(current_episode_steps)
                num_episodes_completed +=1
                
                # Reset for next episode within the rollout collection
                obs_dict, info = env.reset()
                current_obs_grid = obs_dict["grid"]
                current_obs_pieces = obs_dict["pieces"]
                current_episode_rl_reward = 0
                current_episode_steps = 0
                # current_done = False # Will be reset at start of next rollout if this was the last step


        # --- Compute Advantages and Returns ---
        with torch.no_grad():
            last_value_tensor = agent.get_value({"grid": current_obs_grid, "pieces": current_obs_pieces})
        
        rollout_buffer.compute_returns_and_advantages(last_value_tensor, current_done)

        # --- Perform PPO Update ---
        avg_policy_loss, avg_value_loss, avg_entropy = agent.update(rollout_buffer)

        # --- Logging and Progress Bar ---
        avg_rl_score_window = np.mean(episode_rewards_window) if episode_rewards_window else -1
        avg_game_score_window = np.mean(game_scores_window) if game_scores_window else -1
        avg_steps_window = np.mean(steps_window) if steps_window else -1
        
        pbar.set_description(
            f"Update: {update_num}/{num_updates} | AvgRLScore: {avg_rl_score_window:.2f} | "
            f"AvgGameScore: {avg_game_score_window:.1f} | AvgSteps: {avg_steps_window:.1f} | "
            f"P_Loss: {avg_policy_loss:.3f} | V_Loss: {avg_value_loss:.3f} | Entropy: {avg_entropy:.3f}"
        )

        update_data = {
            "update_num": update_num,
            "total_timesteps": total_timesteps_collected,
            "avg_rl_score_window": float(avg_rl_score_window),
            "avg_game_score_window": float(avg_game_score_window),
            "avg_steps_window": float(avg_steps_window),
            "avg_policy_loss": float(avg_policy_loss),
            "avg_value_loss": float(avg_value_loss),
            "avg_entropy": float(avg_entropy),
            "num_episodes_this_update_cycle": num_episodes_completed # Tracks episodes since last log/update
        }
        training_log["updates_data"].append(update_data)
        num_episodes_completed = 0 # Reset for next update cycle

        if update_num % 50 == 0 or update_num == 1: # Log frequency
            with open(json_log_path, 'w') as f:
                json.dump(training_log, f, indent=4)
        
        if update_num % s_ai.VISUALIZE_EVERY_N_UPDATES_PPO == 0 and update_num > 0:
            save_ppo_model(agent, version=update_num)
            # Add visualization logic here if needed, e.g., run one episode with vis_env

    print("PPO Training finished.")
    pbar.close()
    save_ppo_model(agent, filename="BD_PPO_final")
    with open(json_log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"PPO Training log saved to {json_log_path}")
    
    # if vis_env:
    #     vis_env.close()

if __name__ == '__main__':
    train_ppo()