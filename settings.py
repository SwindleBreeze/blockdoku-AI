# settings.py
import pygame

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GRID_COLOR = (50, 50, 50)
EMPTY_CELL_COLOR = (40, 40, 40)

# --- Grid Dimensions ---
GRID_WIDTH = 9 # 9x9 grid
GRID_HEIGHT = 9
TILE_SIZE = 40 # Pixels per grid cell
GRID_LINE_WIDTH = 1
BLOCK_BORDER_WIDTH = 1 # Border around each small square in a piece

# --- Screen Dimensions ---
INFO_AREA_WIDTH = 300 # Width for score and piece selection
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE + INFO_AREA_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + 80
GRID_START_X = 0
GRID_START_Y = 0

# --- Piece Selection Area ---
PIECE_AREA_X = GRID_WIDTH * TILE_SIZE + 25
PIECE_AREA_Y = 40
PIECE_SPACING = 125 # Vertical space between pieces in selection area

# --- Game Settings ---
NUM_PIECES_AVAILABLE = 3
FPS = 60

# --- Piece Definitions ---
# Shapes are defined as lists of (row, col) offsets from an anchor point (usually top-left)
# Colors are defined per piece type
PIECE_DEFINITIONS = {
    # Example pieces (add many more!)
    'dot': {'shape': [(0, 0)], 'color': (255, 0, 0)}, # Red
    'domino_v': {'shape': [(0, 0), (1, 0)], 'color': (0, 255, 0)}, # Green
    'domino_h': {'shape': [(0, 0), (0, 1)], 'color': (0, 0, 255)}, # Blue
    'corner_small': {'shape': [(0, 0), (1, 0), (0, 1)], 'color': (255, 255, 0)}, # Yellow
    'l_shape': {'shape': [(0, 0), (1, 0), (2, 0), (2, 1)], 'color': (0, 255, 255)}, # Cyan
    't_shape': {'shape': [(0,0), (0,1), (0,2), (1,1)], 'color': (255, 0, 255)}, # Magenta
    's_shape': {'shape': [(0,1), (0,2), (1,0), (1,1)], 'color': (255, 165, 0)}, # Orange
    'z_shape': {'shape': [(0,0), (0,1), (1,1), (1,2)], 'color': (128, 0, 128)}, # Purple
    'square': {'shape': [(0,0), (0,1), (1,0), (1,1)], 'color': (0, 128, 0)}, # Dark Green
    'line_3v': {'shape': [(0,0), (1,0), (2,0)], 'color': (255, 192, 203)}, # Pink
    'line_3h': {'shape': [(0,0), (0,1), (0,2)], 'color': (165, 42, 42)}, # Brown
    'u_shape': {'shape': [(0,0), (0,2), (1,0), (1,1), (1,2)], 'color': (75, 0, 130)}, # Indigo
    # Add more shapes: lines of 4/5, larger Ls, squares (3x3), etc.
    'big_square': {'shape': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)], 'color': (218, 165, 32)} # Goldenrod
}

PIECE_KEYS = list(PIECE_DEFINITIONS.keys())
NUM_PIECE_TYPES = len(PIECE_KEYS)
PIECE_KEY_TO_ID = {key: i for i, key in enumerate(PIECE_KEYS)}

# --- Scoring ---
SCORE_PER_BLOCK = 1
SCORE_PER_LINE = 18 * 2 # 9 blocks + 9 bonus
SCORE_PER_SQUARE = 18 * 2 # 9 blocks + 9 bonus
# Add potential combo bonuses later if desired



# --- AI Settings ---
# State representation
STATE_GRID_CHANNELS = 1 # Grayscale grid
STATE_PIECE_VECTOR_SIZE = NUM_PIECE_TYPES # Size of the multi-hot vector
# Add more channels later if encoding pieces directly on grid planes

# Action space: 3 pieces * 9 rows * 9 cols
ACTION_SPACE_SIZE = NUM_PIECES_AVAILABLE * GRID_WIDTH * GRID_HEIGHT # 3 * 9 * 9 = 243

# DQN Hyperparameters (Example Values - Tune these!)
LEARNING_RATE = 5e-5
GAMMA = 0.99 # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 500000  # How many steps to decay epsilon over
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 5000 # Steps between updating target network
LEARNING_STARTS = 5000 # Steps before starting training (fill buffer)

# Training Settings
NUM_EPISODES_TRAIN = 2000
# MAX_STEPS_PER_EPISODE = 1000 # Optional: Limit episode length

# Visualization/Play Settings
VISUALIZE_EVERY_N_EPISODES = 1000 # How often to show a sped-up episode
PLAY_FPS = 10 # FPS for the AI playing visually
TRAIN_VIS_FPS = 120 # Faster FPS for visualized training runs

# Reward Scaling
REWARD_SCALING_FACTOR = 15.0
INVALID_MOVE_PENALTY = -0.5 # Reduced penalty (scaled)

# Gradient Clipping
GRADIENT_CLIP_NORM = 1.0 # Keep clipping enabled

# File Paths
MODEL_SAVE_DIR = "saved_models"