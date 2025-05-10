# play.py
import time
import torch # Use torch
import numpy as np
import os
import pygame # Need pygame for the quit event check
import settings as s
from blockdoku_env import BlockdokuEnv
from ppo_agent import DQNAgent # PyTorch version

def play(model_path):
    # Check for the PyTorch model file extension (.pth or .pt)
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        # Try finding default file if path invalid
        default_model_file = os.path.join(s.MODEL_SAVE_DIR, "BD_v360.pth")
        if os.path.exists(default_model_file):
             print(f"Attempting to load default model: {default_model_file}")
             model_path = default_model_file
        else:
             print("No model file found in default directory either. Cannot play.")
             return

    env = BlockdokuEnv(render_mode="human")
    grid_shape_numpy = (s.GRID_HEIGHT, s.GRID_WIDTH, s.STATE_GRID_CHANNELS)
    piece_vector_size = s.STATE_PIECE_VECTOR_SIZE
    agent = DQNAgent(grid_shape_numpy, piece_vector_size, env.action_size, load_model_path=model_path)
    agent.epsilon = 0.0 # Play greedily

    print(f"Playing using model: {model_path}")

    state_np, info = env.reset() # Get initial numpy state
    done = False
    total_score = 0
    steps = 0

    while not done:
        env.render(fps=s.PLAY_FPS_AI)

        valid_mask = info.get("valid_action_mask", None)
        # Pass numpy state to act method
        action = agent.act(state_np, valid_action_mask=valid_mask, use_epsilon=False)

        next_state_np, reward, done, info = env.step(action)

        state_np = next_state_np # Update numpy state
        total_score += reward
        steps += 1

        print(f"\rStep: {steps} | Action: {action} | Reward: {reward:.2f} | Score: {total_score:.2f}", end="")

        # Check for window close event
        should_quit = False
        for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 print("\nPygame window closed by user.")
                 should_quit = True
                 break
        if should_quit:
             break

    print(f"\n--- Game Over ---")
    print(f"Final Score: {total_score}")
    print(f"Total Steps: {steps}")

    # Keep window open for a bit
    print("Closing window in 5 seconds...")
    start_time = time.time()
    while time.time() - start_time < 5:
        env.render()
        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
                break
        if should_quit:
            break

    env.close()


if __name__ == '__main__':
    # Specify the PyTorch model file (.pth or .pt)
    model_to_play = os.path.join(s.MODEL_SAVE_DIR, "BD_v20000.pth")
    play(model_to_play)