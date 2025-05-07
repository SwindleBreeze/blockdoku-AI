# blockdoku/settings.py

# --- Import fundamental game parameters needed for AI ---
from game.settings import (
    GRID_WIDTH, GRID_HEIGHT, NUM_PIECES_AVAILABLE,
    NUM_PIECE_TYPES, PIECE_KEY_TO_ID, # PIECE_DEFINITIONS, PIECE_KEYS could also be imported if needed for debug/setup
    STATE_GRID_CHANNELS
)

# --- AI Settings ---
# State representation (derived or AI specific)
STATE_PIECE_VECTOR_SIZE = NUM_PIECE_TYPES # Derived from game settings

# Action space (derived)
ACTION_SPACE_SIZE = NUM_PIECES_AVAILABLE * GRID_WIDTH * GRID_HEIGHT

# DQN Hyperparameters (Values from your last provided settings.py, adjust as needed)
LEARNING_RATE = 0.0001      # Adjusted based on previous discussion
GAMMA = 0.99
EPSILON_START = 1.0        # Adjusted
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000 # Adjusted significantly
REPLAY_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 5000   # Adjusted
LEARNING_STARTS = 30000

# Training Settings
NUM_EPISODES_TRAIN = 20000

# Visualization/Play Settings (for AI playing/training vis)
VISUALIZE_EVERY_N_EPISODES = 5000
PLAY_FPS_AI = 30 # FPS for the AI playing visually
TRAIN_VIS_FPS_AI = 120 # Faster FPS for visualized training runs

# --- RL Reward Parameters (New Structure) ---
REWARD_BLOCK_PLACED = -0.25         # Gentler penalty for placing a block
REWARD_LINE_SQUARE_CLEAR = 1.0     # Reward per line or square actually cleared
REWARD_ALMOST_FULL = 0.15          # Bonus for making a line/col/sq 7 or 8 full
INVALID_MOVE_PENALTY = -0.65        # Moderate penalty
STUCK_PENALTY_RL = -0.5            # Moderate penalty if game ends (stuck)


# --- Gradient Clipping ---
GRADIENT_CLIP_NORM = 1.0

# --- File Paths ---
MODEL_SAVE_DIR = "saved_models"

# --- Colors (Only if AI-specific visualization uses them directly from here) ---
# WHITE = (255, 255, 255) # Prefer sourcing from game.settings if used for general display
# BLACK = (0, 0, 0)