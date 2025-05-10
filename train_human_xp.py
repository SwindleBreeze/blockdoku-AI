# train.py
import os
# Fix for MKL threading issues
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
from utils import preprocess_human_data
import numpy as np
import os
import time
from collections import deque
from tqdm import tqdm 
import torch.nn.functional as F
import torch

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

    model_filename = "blockdoku_dqn_torch.pth" 
    latest_model_path = os.path.join(s.MODEL_SAVE_DIR, model_filename) 

    grid_shape_numpy = (s.GRID_HEIGHT, s.GRID_WIDTH, s.STATE_GRID_CHANNELS)
    
    # Initialize a fresh agent
    agent = DQNAgent(grid_shape_numpy, env.action_size, load_model_path=None)
    human_demonstrations = []
    # ---------- HUMAN DEMONSTRATION LEARNING PHASE ----------
    print("Starting human demonstration learning phase...")
    try:
        human_data_path = os.path.join(os.path.dirname(__file__), "human_games", "recorded_human_games.json")
        with open(human_data_path, 'r') as f:
            raw_human_data_log = json.load(f)
        
        print("Preprocessing human gameplay data using environment simulation...")
        # Pass the shared 'env' instance. It will be reset internally by preprocess_human_data for each game.
        processed_human_data = preprocess_human_data(raw_human_data_log, env) 
        human_games_processed = processed_human_data.get("games", [])

        if not human_games_processed:
            print("No human games successfully processed. Skipping demonstration learning.")
        else:
            print(f"Loaded and processed {len(human_games_processed)} human games for demonstration learning.")
            
            human_game_scores = [game.get("final_score", 0) for game in human_games_processed]
            if human_game_scores: # Check if list is not empty
                print(f"Human game scores: min={min(human_game_scores)}, max={max(human_game_scores)}, avg={np.mean(human_game_scores):.1f}")

            num_pretrain_epochs = 10 
            human_batch_size = 8 

            # Collect all processed (s, a, r, s', d) transitions from all games
            all_experiences = []
            for game in human_games_processed:
                all_experiences.extend(game.get("moves", [])) # Each 'move' is now a dict with state, action, reward, next_state, done

            if not all_experiences:
                print("No experiences extracted from human games. Skipping pre-training.")
            else:
                print(f"Pre-training on {len(all_experiences)} human experiences for {num_pretrain_epochs} epochs...")

                for epoch in range(num_pretrain_epochs):
                    import random
                    random.shuffle(all_experiences)
                    
                    total_loss = 0
                    experiences_processed_in_epoch = 0
                    
                    batches = [all_experiences[i:i + human_batch_size] for i in range(0, len(all_experiences), human_batch_size)]
                    
                    for batch_experiences in tqdm(batches, desc=f"Epoch {epoch+1}/{num_pretrain_epochs}"):
                        if not batch_experiences:
                            continue
                        
                        batch_grids = []
                        batch_pieces_spatial = []
                        batch_actions = []
                        
                        for exp in batch_experiences:
                            # exp is a dict: {"state": s_dict, "action": a, "reward": r, "next_state": ns_dict, "done": d}
                            state_dict = exp.get("state")
                            action = exp.get("action")
                            
                            if state_dict and isinstance(state_dict.get("grid"), np.ndarray) and \
                            isinstance(state_dict.get("pieces_spatial"), np.ndarray) and action is not None:
                                
                                batch_grids.append(state_dict["grid"])
                                batch_pieces_spatial.append(state_dict["pieces_spatial"])
                                batch_actions.append(action)
                                
                                # Add this high-quality human experience to the replay buffer
                                agent.buffer.add(
                                    state_dict, 
                                    action, 
                                    exp.get("reward"), 
                                    exp.get("next_state"), 
                                    exp.get("done")
                                )
                            else:
                                print(f"Warning: Incomplete experience data in batch: {exp}")
                                continue

                        if not batch_grids: # If all experiences in batch were invalid
                            continue
                            
                        try:
                            grid_tensor = torch.from_numpy(np.stack(batch_grids)).float().permute(0, 3, 1, 2).to(agent.device)
                            pieces_spatial_tensor = torch.from_numpy(np.stack(batch_pieces_spatial)).float().to(agent.device)
                            action_tensor = torch.tensor(batch_actions).long().to(agent.device)
                            
                            agent.model.train()
                            pred_q_values = agent.model(grid_tensor, pieces_spatial_tensor) # Model predicts Q-values for all actions
                            
                            # For supervised learning with CrossEntropyLoss, target is the action index itself
                            loss = F.cross_entropy(pred_q_values, action_tensor)
                            
                            agent.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=s.GRADIENT_CLIP_NORM)
                            agent.optimizer.step()
                            
                            total_loss += loss.item() * len(batch_actions)
                            experiences_processed_in_epoch += len(batch_actions)
                        except Exception as e:
                            print(f"Error in batch forward/backward for human data: {e}")
                            continue
                    
                    agent.update_target_network() # Good to do after each epoch of pre-training
                    
                    avg_loss = total_loss / experiences_processed_in_epoch if experiences_processed_in_epoch > 0 else 0
                    print(f"Epoch {epoch+1}/{num_pretrain_epochs} - Avg Loss: {avg_loss:.4f}, Processed: {experiences_processed_in_epoch}/{len(all_experiences)} experiences")

                print("Human demonstration learning phase complete!")
                pretrained_model_path = os.path.join(s.MODEL_SAVE_DIR, "blockdoku_pretrained_with_human_actions.pth")
                agent.save(pretrained_model_path)
                print(f"Pre-trained model saved to {pretrained_model_path}")

    except Exception as e:
        print(f"Error during human demonstration learning: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("Continuing with standard training...")

    # ---------- STANDARD RL TRAINING PHASE ----------
    # Rest of your regular training code follows...
    episode_rewards_window = deque(maxlen=100)
    game_scores_window = deque(maxlen=100)
    losses_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    json_log_path = f"logs/training_log_{timestamp}.json"
    os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
    
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
            "stuck_penalty_rl": s.STUCK_PENALTY_RL,
            "used_human_demos": True  # Add flag to indicate human demo usage
        }
    }

    # Skip pre-filling if we already have data from human demos
    if len(agent.buffer) < s.LEARNING_STARTS:
        print(f"Pre-filling replay buffer to {s.LEARNING_STARTS} experiences...")
        buffer_fill_start_time = time.time()
        buffer_steps = 0
        buffer_episode = 0
        
        # Track ALL episode statistics
        all_rewards = []
        all_game_scores = []
        all_steps = []
        
        pbar = tqdm(total=s.LEARNING_STARTS - len(agent.buffer), bar_format='{l_bar}{bar}| {percentage:3.0f}%')
        
        while len(agent.buffer) < s.LEARNING_STARTS:
            # Rest of your buffer filling code...
            buffer_episode += 1
            state_np, info = env.reset()
            current_episode_reward = 0
            steps_in_episode = 0
            done = False
            
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
                
                if len(agent.buffer) >= s.LEARNING_STARTS:
                    break
            
            all_rewards.append(current_episode_reward)
            all_game_scores.append(info.get("game_score_display", 0))
            all_steps.append(steps_in_episode)
        
        pbar.close()
        
        avg_reward_all = np.mean(all_rewards)
        avg_game_score_all = np.mean(all_game_scores)
        avg_steps_all = np.mean(all_steps)
        
        buffer_fill_time = time.time() - buffer_fill_start_time
        print(f"\nBuffer filled with {buffer_steps} additional experiences over {buffer_episode} episodes in {buffer_fill_time:.2f} seconds")
        print(f"All episodes statistics:")
        print(f"  RL Score: avg={avg_reward_all:.2f}")
        print(f"  Game Score: avg={avg_game_score_all:.2f}")
        print(f"  Steps per episode: avg={avg_steps_all:.1f}")
    
    print(f"Experience buffer status:")
    print(f"  Current buffer size: {len(agent.buffer)} experiences")
    
    # Main training loop
    print("Starting RL Training...")
    pbar = tqdm(range(1, s.NUM_EPISODES_TRAIN + 1), initial=1, total=s.NUM_EPISODES_TRAIN, unit="ep")
    
    for episode in pbar:
        # Rest of your training loop...
        # This remains largely unchanged
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
            loss = agent.replay()
            
            state_np = next_state_np
            current_episode_rl_reward += rl_reward_step
            
            if loss is not None and loss > 0:
                total_episode_loss += loss
                learn_steps_in_episode += 1
            steps_in_episode += 1

        # Rest of episode handling code...
        episode_rewards_window.append(current_episode_rl_reward)
        game_scores_window.append(info.get("game_score_display", 0))
        if learn_steps_in_episode > 0:
            losses_window.append(total_episode_loss / learn_steps_in_episode)
        steps_window.append(steps_in_episode)

        avg_rl_reward = np.mean(episode_rewards_window)
        avg_game_score = np.mean(game_scores_window)
        avg_loss = np.mean(losses_window) if losses_window else 0
        avg_steps = np.mean(steps_window)
        
        episode_time_taken = time.time() - episode_start_time

        # Update progress bar
        pbar.set_description(
            f"RLS: {avg_rl_reward:.2f} | GS: {avg_game_score:.1f} | S: {avg_steps:.1f} | L: {avg_loss:.4f} | E: {agent.epsilon:.2f}"
        )

        # Record episode data
        episode_data = {
            "episode": episode,
            "rl_score": float(current_episode_rl_reward),
            "avg_rl_score": float(avg_rl_reward),
            "game_score": int(info.get("game_score_display", 0)),
            "avg_game_score": float(avg_game_score),
            "avg_loss": float(avg_loss) if avg_loss else 0.0,
            "epsilon": float(agent.epsilon),
            "steps": steps_in_episode,
            "buffer_size": len(agent.buffer),
            "time_taken": float(episode_time_taken),
            "lines_cleared": info.get("lines_cleared", 0),
            "cols_cleared": info.get("cols_cleared", 0),
            "squares_cleared": info.get("squares_cleared", 0),
            "almost_full_count": info.get("almost_full_count", 0)
        }
        training_log["episodes"].append(episode_data)
        
        if episode % 50 == 0 or episode == 1:
            with open(json_log_path, 'w') as f:
                json.dump(training_log, f, indent=4)
         
        if episode % s.VISUALIZE_EVERY_N_EPISODES == 0 and episode > 0:
            save_model(agent, version=episode)

        if episode % s.SCH_UPDATE == 0:
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

if __name__ == '__main__':
    train()