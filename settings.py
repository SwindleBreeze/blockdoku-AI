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
LEARNING_RATE = 0.0005      # Step size for gradient updates (controls how quickly model parameters change)
WEIGHT_DECAY = 0.00001       # L2 regularization term (helps prevent overfitting)
GAMMA = 0.99                # Discount factor for future rewards (close to 1 = long-term focus)
GRADIENT_CLIP_NORM = 1.0    # Max gradient magnitude to prevent exploding gradients
# GRADIENT_CLIP_NORM = 1.0 #OG

# SCGEDULER 
SCH_UPDATE = 1000 # How often to update the model (in steps)
SCH_GAMA = 1.0 # How much to decay the learning rate (0.1 = 10% decay) 
SCH_MILESTONE= [3000, 7000, 10000, 13000, 15000, 17000]

# Exploration strategy
EPSILON_START = 0.10         # Initial exploration rate (100% random actions)
EPSILON_END = 0.05          # Final exploration rate (5% random actions)
EPSILON_DECAY_STEPS = 75_000 # Steps over which epsilon decays from start to end value
POWER_DECAY = 2.0           # Power for polynomial decay (higher = more gradual initial decay)
BUMP_PERIOD = 3000          # Cycle length for periodic exploration bumps after decay
BUMP_SIZE = 0.05            # Magnitude of periodic exploration bumps (0.1 = up to 10% bump)

# Memory and batch processing
REPLAY_BUFFER_SIZE = 300_000  # Maximum experiences to store (3M transitions)
BATCH_SIZE = 512            # Number of experiences to learn from in each update
LEARNING_STARTS = 1_000    # Steps to collect before starting to learn (builds initial experience)

# Target network
TARGET_UPDATE_FREQ = 100   # How often to updatne target network (in steps)

# Training Settings / epoh 
NUM_EPISODES_TRAIN = 100000

# Visualization/Play Settings (for AI playing/training vis)
VISUALIZE_EVERY_N_EPISODES = 2500
PLAY_FPS_AI = 30 # FPS for the AI playing visually
TRAIN_VIS_FPS_AI = 120 # Faster FPS for visualized training runs

# --- RL Reward Parameters (New Structure) ---
REWARD_BLOCK_PLACED = -0.1         # Gentler penalty for placing a block
REWARD_LINE_SQUARE_CLEAR = 2.0     # Reward per line or square actually cleared
REWARD_ALMOST_FULL = 0.5          # Bonus for making a line/col/sq 7 or 8 full
INVALID_MOVE_PENALTY = -0.65        # Moderate penalty
STUCK_PENALTY_RL = -4.0            # Moderate penalty if game ends (stuck)

# --- Prioritized Experience Replay (PER) Settings ---
PER_ENABLED = True # Set to False to easily revert to standard buffer if needed
PER_ALPHA = 0.5           # Priority exponent (0=uniform, 1=full priority) 
PER_BETA_START = 0.4       # Initial importance sampling exponent
PER_BETA_END = 1.0         # Final importance sampling exponent (annealed towards this)
PER_BETA_ANNEALING_STEPS = 35_000 # Steps over which beta anneals (adjust based on total training steps)
PER_EPSILON = 0.1       # Small value added to priorities to ensure non-zero probability


# --- File Paths ---
MODEL_SAVE_DIR = "saved_models"

# --- Colors (Only if AI-specific visualization uses them directly from here) ---
# WHITE = (255, 255, 255) # Prefer sourcing from game.settings if used for general display
# BLACK = (0, 0, 0)