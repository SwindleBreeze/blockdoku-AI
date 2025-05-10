# train.py
import os
# Fix for MKL threading issues
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm 

import settings as s
from blockdoku_env import BlockdokuEnv
from dqn_agent import DQNAgent 
import json
import datetime

    
def save_model(agent, filename="BD", version=None, save_dir="saved_models"):
    model_dir = save_dir if save_dir else s.MODEL_SAVE_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = filename if filename else "blockdoku_dqn_torch"
    if version is not None:
        base_name = f"{base_name}_v{version}"
    model_path = os.path.join(model_dir, f"{base_name}.pth")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def train():
    if not os.path.exists(s.MODEL_SAVE_DIR):
        os.makedirs(s.MODEL_SAVE_DIR)


    env = BlockdokuEnv(render_mode=None)
    # vis_env = None # Not strictly needed if visualization part is commented out

    model_filename = "blockdoku_dqn_torch.pth" 
    latest_model_path = os.path.join(s.MODEL_SAVE_DIR, model_filename) 

    grid_shape_numpy = (s.GRID_HEIGHT, s.GRID_WIDTH, s.STATE_GRID_CHANNELS)
    # piece_vector_size = s.STATE_PIECE_VECTOR_SIZE
    
    # Ensure you are not loading a model that might be incompatible with new reward structure
    # For a fresh start, ensure latest_model_path doesn't exist or pass None
    agent = DQNAgent(grid_shape_numpy, env.action_size,
                     load_model_path=None) # Start fresh: load_model_path=None
                     # load_model_path=latest_model_path if os.path.exists(latest_model_path) else None)


    episode_rewards_window = deque(maxlen=100) # For RL rewards
    game_scores_window = deque(maxlen=100)     # For simple game scores
    losses_window = deque(maxlen=100)          # Renamed for clarity
    steps_window = deque(maxlen=100)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    json_log_path = f"logs/training_log_{timestamp}.json"
    os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
    
    # Store AI settings used for this run in the log
    training_log = {
        "episodes": [],
        "settings": {
            "num_episodes": s.NUM_EPISODES_TRAIN,
            "epsilon_start": s.EPSILON_START,
            "epsilon_end": s.EPSILON_END,
            "epsilon_decay_steps": s.EPSILON_DECAY_STEPS,
            "learning_rate": s.LEARNING_RATE,
            "gamma": s.GAMMA,
            "batch_size": s.BATCH_SIZE,
            "target_update_freq": s.TARGET_UPDATE_FREQ,
            "learning_starts": s.LEARNING_STARTS,
            "replay_buffer_size": s.REPLAY_BUFFER_SIZE,
            "reward_block_placed": s.REWARD_BLOCK_PLACED,
            "reward_line_square_clear": s.REWARD_LINE_SQUARE_CLEAR,
            "invalid_move_penalty": s.INVALID_MOVE_PENALTY,
            "stuck_penalty_rl": s.STUCK_PENALTY_RL
        }
    }

    
    # Pre-fill buffer with random episodes
    print(f"Pre-filling replay buffer to {s.LEARNING_STARTS} experiences...")
    buffer_fill_start_time = time.time()
    buffer_steps = 0
    buffer_episode = 0

    # Track ALL episode statistics (not just sliding window)
    all_rewards = []
    all_game_scores = []
    all_steps = []

    # Simple progress bar that only shows percent
    pbar = tqdm(total=s.LEARNING_STARTS, bar_format='{l_bar}{bar}| {percentage:3.0f}%')

    while buffer_steps < s.LEARNING_STARTS:
        buffer_episode += 1
        state_np, info = env.reset()
        current_episode_reward = 0
        steps_in_episode = 0
        done = False
        
        # Play a complete episode
        while not done:
            valid_mask = info.get("valid_action_mask", None)
            action = agent.act(state_np, valid_action_mask=valid_mask, use_epsilon=True)
            
            next_state_np, reward, done, info = env.step(action)
            agent.remember(state_np, action, reward, next_state_np, done)
            
            state_np = next_state_np
            current_episode_reward += reward
            steps_in_episode += 1
            buffer_steps += 1
            pbar.update(1)
            
            # Stop if we've reached the target steps
            if buffer_steps >= s.LEARNING_STARTS:
                break
        
        # Track statistics for EVERY episode
        all_rewards.append(current_episode_reward)
        all_game_scores.append(info.get("game_score_display", 0))
        all_steps.append(steps_in_episode)

    pbar.close()

    # Calculate averages across ALL episodes
    avg_reward_all = np.mean(all_rewards)
    avg_game_score_all = np.mean(all_game_scores)
    avg_steps_all = np.mean(all_steps)

    # Buffer filling complete
    buffer_fill_time = time.time() - buffer_fill_start_time
    print(f"\nBuffer filled with {buffer_steps} experiences over {buffer_episode} episodes in {buffer_fill_time:.2f} seconds")
    print(f"All episodes statistics:")
    print(f"  RL Score: avg={avg_reward_all:.2f}")
    print(f"  Game Score: avg={avg_game_score_all:.2f}")
    print(f"  Steps per episode: avg={avg_steps_all:.1f}")
    print(f"Experience buffer status:")
    print(f"  Current buffer size: {len(agent.buffer)} experiences")
    # Main training loop begins here
    print("Starting Training...")
    pbar = tqdm(range(1, s.NUM_EPISODES_TRAIN + 1), initial=1, total=s.NUM_EPISODES_TRAIN, unit="ep")
    
    for episode in pbar:
        state_np, info = env.reset()
        current_episode_rl_reward = 0
        total_episode_loss = 0
        learn_steps_in_episode = 0
        steps_in_episode = 0
        done = False
        
        episode_start_time = time.time()

        while not done:
            valid_mask = info.get("valid_action_mask", None)
            action = agent.act(state_np, valid_action_mask=valid_mask, use_epsilon=True)
            
            next_state_np, rl_reward_step, done, info = env.step(action)
            
            agent.remember(state_np, action, rl_reward_step, next_state_np, done)
            # Now we can start learning right away since the buffer is filled
            loss = agent.replay()
            
            state_np = next_state_np
            current_episode_rl_reward += rl_reward_step
            
            if loss is not None and loss > 0:
                total_episode_loss += loss
                learn_steps_in_episode += 1
            steps_in_episode += 1

        episode_rewards_window.append(current_episode_rl_reward)
        game_scores_window.append(info.get("game_score_display", 0)) # Get game score from info
        if learn_steps_in_episode > 0:
            losses_window.append(total_episode_loss / learn_steps_in_episode)
        steps_window.append(steps_in_episode)

        avg_rl_reward = np.mean(episode_rewards_window) if episode_rewards_window else 0
        avg_game_score = np.mean(game_scores_window) if game_scores_window else 0
        avg_loss = np.mean(losses_window) if losses_window else 0
        avg_steps = np.mean(steps_window) if steps_window else 0
        
        episode_time_taken = time.time() - episode_start_time

        pbar.set_description(
            f"RLS: {avg_rl_reward:.2f} | GS: {avg_game_score:.1f} | S: {avg_steps:.1f} | L: {avg_loss:.4f} | E: {agent.epsilon:.2f}"
        )

        episode_data = {
            "episode": episode,
            "rl_score": float(current_episode_rl_reward), # RL score
            "avg_rl_score": float(avg_rl_reward),
            "game_score": int(info.get("game_score_display", 0)), # Game score from info
            "avg_game_score": float(avg_game_score),
            "avg_loss": float(avg_loss) if avg_loss else 0.0,
            "epsilon": float(agent.epsilon),
            "steps": steps_in_episode,
            "buffer_size": len(agent.buffer),
            "time_taken": float(episode_time_taken),
            "lines_cleared": info.get("lines_cleared", 0),
            "cols_cleared": info.get("cols_cleared", 0),
            "squares_cleared": info.get("squares_cleared", 0),
            "almost_full_count": info.get("almost_full_count", 0) # New

        }
        training_log["episodes"].append(episode_data)
        
        if episode % 50 == 0 or episode == 1:
            with open(json_log_path, 'w') as f:
                json.dump(training_log, f, indent=4) # Added indent for readability
         
        if episode % s.VISUALIZE_EVERY_N_EPISODES == 0 and episode > 0:
            save_model(agent, version=episode) 
            # Visualization part can be uncommented if vis_env is set up
            # print("\n--- Running Visualized Episode ---")
            # if vis_env is None:
            #      vis_env = BlockdokuEnv(render_mode="human")
            # ... (visualization loop) ...

        if episode % s.SCH_UPDATE == 0:  # Update every 100 episodes
            current_lr = agent.scheduler_step()
            if current_lr is not None:
                episode_data["learning_rate"] = float(current_lr)

    print("Training finished.")
    pbar.close() 
    agent.save(latest_model_path) 
    print(f"Final model saved to {latest_model_path}")

    with open(json_log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"Training log saved to {json_log_path}")
    
    # if vis_env: # Ensure vis_env is closed if it was used
    #     vis_env.close()
        
if __name__ == '__main__':
    train()