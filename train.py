# train.py
import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm # Import tqdm

import settings as s
from blockdoku_env import BlockdokuEnv
from dqn_agent import DQNAgent # PyTorch version

# --- PyTorch TensorBoard Logging ---
try:
    from torch.utils.tensorboard import SummaryWriter
    log_dir = "logs/dqn_torch/" + time.strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging TensorBoard to {log_dir}")
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("TensorBoard for PyTorch not found. Install with 'pip install tensorboard'. Logging disabled.")
    summary_writer = None
    TENSORBOARD_AVAILABLE = False
# ---------------------------------

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
    start_time = time.time()

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

        while not done:
            # ... (inner loop logic: act, step, remember, replay) ...
            valid_mask = info.get("valid_action_mask", None)
            action = agent.act(state_np, valid_action_mask=valid_mask, use_epsilon=True)
            next_state_np, reward, done, info = env.step(action)
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
        pbar.set_description(
            f"Avg Score: {avg_score:.2f} | Last Score: {episode_score:.2f} | Avg Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.3f}"
        )

        # --- Log to TensorBoard (less frequently than tqdm update) ---
        if TENSORBOARD_AVAILABLE and summary_writer:
            summary_writer.add_scalar('Score', episode_score, episode)
            summary_writer.add_scalar('Average Score (100 ep)', avg_score, episode)
            summary_writer.add_scalar('Average Loss (100 ep)', avg_loss, episode)
            summary_writer.add_scalar('Epsilon', agent.epsilon, episode)
            summary_writer.add_scalar('Episode Steps', steps, episode)
            summary_writer.add_scalar('Buffer Size', len(agent.buffer), episode)

        # --- Optional: Print full summary less often ---
        if episode % 100 == 0: # Print a summary line every 100 episodes
             elapsed_time = time.time() - start_time
             print(f"\nEp {episode} Summary | Avg Score: {avg_score:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed_time:.1f}s")


        # Visualized Training Run (no changes needed here)
        if episode % s.VISUALIZE_EVERY_N_EPISODES == 0:
            print("\n--- Running Visualized Episode ---")
            if vis_env is None:
                 vis_env = BlockdokuEnv(render_mode="human")

            vis_state_np, vis_info = vis_env.reset()
            vis_done = False
            vis_score = 0
            while not vis_done:
                 vis_env.render(fps=s.TRAIN_VIS_FPS)
                 valid_mask = vis_info.get("valid_action_mask", None)
                 # Pass numpy state to agent's act method
                 vis_action = agent.act(vis_state_np, valid_action_mask=valid_mask, use_epsilon=False)
                 vis_next_state_np, vis_reward, vis_done, vis_info = vis_env.step(vis_action)
                 vis_state_np = vis_next_state_np # Update numpy state
                 vis_score += vis_reward
            print(f"--- Visualized Score: {vis_score:.2f} ---\n")

    print("Training finished.")
    pbar.close() # Close the tqdm progress bar
    agent.save(latest_model_path) # Save final model
    print(f"Final model saved to {latest_model_path}")

    if vis_env:
        vis_env.close()
    if TENSORBOARD_AVAILABLE and summary_writer:
        summary_writer.close() # Close the TensorBoard writer

if __name__ == '__main__':
    # No specific GPU setup needed for PyTorch typically, it handles it automatically
    train()