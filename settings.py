# blockdoku/settings.py

# --- Import fundamental game parameters needed for AI ---
from game.settings import (
    GRID_WIDTH, GRID_HEIGHT, NUM_PIECES_AVAILABLE,
    NUM_PIECE_TYPES, PIECE_KEY_TO_ID,
    STATE_GRID_CHANNELS
)

# --- AI Settings ---
STATE_PIECE_VECTOR_SIZE = NUM_PIECE_TYPES
ACTION_SPACE_SIZE = NUM_PIECES_AVAILABLE * GRID_WIDTH * GRID_HEIGHT

# === DQN Hyperparameters (Keep for reference or if you switch back) ===
# LEARNING_RATE_DQN = 0.0001
# GAMMA_DQN = 0.99
# EPSILON_START = 1.0
# EPSILON_END = 0.15
# EPSILON_DECAY_STEPS = 50_000
# REPLAY_BUFFER_SIZE = 2_000_000
# BATCH_SIZE_DQN = 64
# TARGET_UPDATE_FREQ = 5000
# LEARNING_STARTS = 300_000
# PER_ENABLED = True
# PER_ALPHA = 0.6
# PER_BETA_START = 0.4
# PER_BETA_END = 1.0
# PER_BETA_ANNEALING_STEPS = 250_000
# PER_EPSILON = 1e-8

# === PPO Hyperparameters ===
PPO_LEARNING_RATE = 0.0001       # Learning rate for actor and critic
PPO_GAMMA = 0.99               # Discount factor for rewards
PPO_GAE_LAMBDA = 0.95          # Lambda for Generalized Advantage Estimation
PPO_CLIP_EPSILON = 0.2         # Epsilon for policy loss clipping
PPO_EPOCHS = 15                # Number of epochs to train on collected data
PPO_MINI_BATCH_SIZE = 64       # Mini-batch size for updates
PPO_VALUE_LOSS_COEF = 0.5      # Coefficient for value loss
PPO_ENTROPY_COEF = 0.03        # Coefficient for entropy bonus (encourages exploration)
PPO_NUM_STEPS_PER_UPDATE = 1024 # 2048 # Number of steps to collect before an update
PPO_MAX_GRAD_NORM = 0.5        # Gradient clipping for PPO

# Training Settings
NUM_TOTAL_TIMESTEPS_PPO =  400_000 # 5_000_000 # Example: Total steps for PPO training
# NUM_EPISODES_TRAIN = 400_000 # (from your DQN settings)

# Visualization/Play Settings
VISUALIZE_EVERY_N_UPDATES_PPO = 100 # Changed from episodes to updates for PPO
PLAY_FPS_AI = 30
TRAIN_VIS_FPS_AI = 120

# --- RL Reward Parameters (Can be reused by PPO) ---
REWARD_BLOCK_PLACED = 0.0
REWARD_LINE_SQUARE_CLEAR = 2.0
REWARD_ALMOST_FULL = 0.3
INVALID_MOVE_PENALTY = -0.65
STUCK_PENALTY_RL = -1.0 # This was -2 in your last settings

# Gradient Clipping (GRADIENT_CLIP_NORM is now PPO_MAX_GRAD_NORM for PPO)
# GRADIENT_CLIP_NORM = 1.0 # (from your DQN settings)

# File Paths
MODEL_SAVE_DIR = "saved_models"