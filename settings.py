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
LEARNING_RATE = 0.0002      # Step size for gradient updates (controls how quickly model parameters change)
WEIGHT_DECAY = 0.00001       # L2 regularization term (helps prevent overfitting)
GAMMA = 0.99                # Discount factor for future rewards (close to 1 = long-term focus)
GRADIENT_CLIP_NORM = 0.5    # Max gradient magnitude to prevent exploding gradients
# GRADIENT_CLIP_NORM = 1.0 #OG


# Exploration strategy
EPSILON_START = 1.0         # Initial exploration rate (100% random actions)
EPSILON_END = 0.02          # Final exploration rate (5% random actions)
EPSILON_DECAY_STEPS = 100000 # Steps over which epsilon decays from start to end value
POWER_DECAY = 2.0           # Power for polynomial decay (higher = more gradual initial decay)
BUMP_PERIOD = 3000          # Cycle length for periodic exploration bumps after decay
BUMP_SIZE = 0.0             # Magnitude of periodic exploration bumps (0.1 = up to 10% bump)

# Memory and batch processing
REPLAY_BUFFER_SIZE = 3_000_000  # Maximum experiences to store (3M transitions)
BATCH_SIZE = 512            # Number of experiences to learn from in each update
LEARNING_STARTS = 400000    # Steps to collect before starting to learn (builds initial experience)

# Target network
TARGET_UPDATE_FREQ = 500   # How often to updatne target network (in steps)

# Training Settings / epoh 
NUM_EPISODES_TRAIN = 40000

# Visualization/Play Settings (for AI playing/training vis)
VISUALIZE_EVERY_N_EPISODES = 2500
PLAY_FPS_AI = 30 # FPS for the AI playing visually
TRAIN_VIS_FPS_AI = 120 # Faster FPS for visualized training runs

# --- RL Reward Parameters (New Structure) ---
REWARD_BLOCK_PLACED = -0.25         # Gentler penalty for placing a block
REWARD_LINE_SQUARE_CLEAR = 1.0     # Reward per line or square actually cleared
REWARD_ALMOST_FULL = 0.15          # Bonus for making a line/col/sq 7 or 8 full
INVALID_MOVE_PENALTY = -0.65        # Moderate penalty
STUCK_PENALTY_RL = -1.0            # Moderate penalty if game ends (stuck)




# --- File Paths ---
MODEL_SAVE_DIR = "saved_models"

# --- Colors (Only if AI-specific visualization uses them directly from here) ---
# WHITE = (255, 255, 255) # Prefer sourcing from game.settings if used for general display
# BLACK = (0, 0, 0)