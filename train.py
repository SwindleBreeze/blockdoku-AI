# train.py
import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm # Import tqdm

import settings as s
from blockdoku_env import BlockdokuEnv
from dqn_agent import DQNAgent # PyTorch version
import json
import datetime

    
def save_model(agent, filename="BD", version=None, save_dir="saved_models"):
    # Use provided directory or default from settings
    model_dir = save_dir if save_dir else s.MODEL_SAVE_DIR
    
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Build the filename
    base_name = filename if filename else "blockdoku_dqn_torch"
    if version is not None:
        base_name = f"{base_name}_v{version}"
    
    model_path = os.path.join(model_dir, f"{base_name}.pth")
    # Save the model
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model_path


def train():
    if not os.path.exists(s.MODEL_SAVE_DIR):
        os.makedirs(s.MODEL_SAVE_DIR)

    env = BlockdokuEnv(render_mode=None)
    vis_env = None

    # Construct model path for PyTorch (.pth or .pt extension is common)
    model_filename = "blockdoku_dqn_torch.pth" # Choose an extension
    latest_model_path = os.path.join(s.MODEL_SAVE_DIR, model_filename) # Check for this specific file

    grid_shape_numpy = (s.GRID_HEIGHT, s.GRID_WIDTH, s.STATE_GRID_CHANNELS)
    piece_vector_size = s.STATE_PIECE_VECTOR_SIZE
    # Pass numpy shape (H, W, C) to agent, it handles internal PyTorch shape
    agent = DQNAgent(grid_shape_numpy, piece_vector_size, env.action_size,
                     load_model_path=latest_model_path if os.path.exists(latest_model_path) else None)

    scores_window = deque(maxlen=100)
    losses = deque(maxlen=100)
    

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    json_log_path = f"logs/training_log_{timestamp}.json"
    os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
    training_log = {
        "episodes": [],
        "settings": {
            "num_episodes": s.NUM_EPISODES_TRAIN,
            "epsilon_start": s.EPSILON_START,
            "epsilon_end": s.EPSILON_END,
            "learning_rate": s.LEARNING_RATE,
            # Add other relevant settings
        }
    }

    print("Starting Training...")
    # --- Wrap the main loop with tqdm ---
    # Use initial=1 because range starts from 1, total is the number of episodes
    pbar = tqdm(range(1, s.NUM_EPISODES_TRAIN + 1), initial=1, total=s.NUM_EPISODES_TRAIN, unit="ep")

    for episode in pbar: # Iterate through the tqdm progress bar
        state_np, info = env.reset()
        episode_score = 0
        total_episode_loss = 0
        learn_steps = 0
        steps = 0
        done = False
        
        episode_time = time.time()# time fo a single episode
        steps_window = deque(maxlen=100)
        while not done:
            # ... (inner loop logic: act, step, remember, replay) ...
            valid_mask = info.get("valid_action_mask", None)
            action = agent.act(state_np, valid_action_mask=valid_mask, use_epsilon=True)
            next_state_np, reward, done, info = env.step(action) #
            agent.remember(state_np, action, reward, next_state_np, done)
            loss = agent.replay()

            state_np = next_state_np
            episode_score += reward
            if loss > 0:
                total_episode_loss += loss
                learn_steps += 1
            steps += 1

        scores_window.append(episode_score)
        if learn_steps > 0:
            losses.append(total_episode_loss / learn_steps)

        avg_score = np.mean(scores_window) if scores_window else 0
        avg_loss = np.mean(losses) if losses else 0

        # --- Update tqdm description instead of printing every time ---
        # pbar.set_description(
        #     f"Avg Score: {avg_score:.2f} | Last Score: {info["score"]:.2f} | Avg Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.3f}"
        # )
        steps_window.append(steps)
        elapsed_time = time.time() - episode_time
        avg_steps = np.mean(steps_window) if steps_window else 0
        pbar.set_description(
            f"Avg Score: {avg_score:.2f} | Avg Steps: {avg_steps:.1f} | Avg Loss: {avg_loss:.4f} "
        )

        # --- Save episode data to JSON log ---
        episode_data = {
            "episode": episode,
            "score": float(episode_score),
            "avg_score": float(avg_score),
            "avg_loss": float(avg_loss) if avg_loss else 0.0,
            "epsilon": float(agent.epsilon),
            "steps": steps,
            "buffer_size": len(agent.buffer),
            "game score": int(env.game.score),
            "time_taken": float(elapsed_time)
        }
        training_log["episodes"].append(episode_data)
        # Write to JSON file periodically to avoid loss of data if training crashes
        if episode % 50 == 0 or episode == 1:  # Save every 10 episodes and first episode
            with open(json_log_path, 'w') as f:
                json.dump(training_log, f)
         

        # Visualized Training Run (no changes needed here)
        if episode % s.VISUALIZE_EVERY_N_EPISODES == 0:
            save_model(agent, version=episode) # Save the model
            # print("\n--- Running Visualized Episode ---")
            # if vis_env is None:
            #      vis_env = BlockdokuEnv(render_mode="human")

            # vis_state_np, vis_info = vis_env.reset()
            # vis_done = False
            # vis_score = 0
            # while not vis_done:
            #      vis_env.render(fps=s.TRAIN_VIS_FPS)
            #      valid_mask = vis_info.get("valid_action_mask", None)
            #      # Pass numpy state to agent's act method
            #      vis_action = agent.act(vis_state_np, valid_action_mask=valid_mask, use_epsilon=False)
            #      vis_next_state_np, vis_reward, vis_done, vis_info = vis_env.step(vis_action)
            #      vis_state_np = vis_next_state_np # Update numpy state
            #      vis_score += vis_reward
            # print(f"--- Visualized Score: {vis_score:.2f} ---\n")

    print("Training finished.")
    pbar.close() # Close the tqdm progress bar
    agent.save(latest_model_path) # Save final model
    print(f"Final model saved to {latest_model_path}")

    # Save the training log to JSON
    with open(json_log_path, 'w') as f:
        json.dump(training_log, f)
    print(f"Training log saved to {json_log_path}")
    
    if vis_env:
        vis_env.close()
if __name__ == '__main__':
    # No specific GPU setup needed for PyTorch typically, it handles it automatically
    train()