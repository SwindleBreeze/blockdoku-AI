# settings.py (Your AI settings file)

# --- Import fundamental game parameters needed for AI ---
try:
    from game.settings import (
        GRID_WIDTH, GRID_HEIGHT, NUM_PIECES_AVAILABLE,
        NUM_PIECE_TYPES, PIECE_KEY_TO_ID, STATE_GRID_CHANNELS
    )
except ImportError:
    print("WARNING: Could not import from game.settings. Ensure paths are correct or define game constants manually.")
    # Define fallbacks or raise error if critical
    GRID_WIDTH = 9
    GRID_HEIGHT = 9
    NUM_PIECES_AVAILABLE = 3
    NUM_PIECE_TYPES = 12 # Example, adjust to your game's PIECE_DEFINITIONS
    PIECE_KEY_TO_ID = {} # Populate this based on your game.settings.PIECE_KEYS
    STATE_GRID_CHANNELS = 1


# --- AI General Settings ---
ACTION_SPACE_SIZE = NUM_PIECES_AVAILABLE * GRID_WIDTH * GRID_HEIGHT

# --- RL Reward Parameters (Used by GA's heuristic evaluation now) ---
R_PLACED_BLOCK_IMMEDIATE = 1.0       # Immediate reward for placing a block (can be small, or 0 if covered by game score)
R_CLEARED_LINE_COL_IMMEDIATE = 10.0  # Immediate reward per line or column cleared
R_CLEARED_SQUARE_IMMEDIATE = 10.0    # Immediate reward per 3x3 square cleared
R_ALMOST_FULL_IMMEDIATE = 3.0        # Immediate reward for creating an "almost full" region
R_GAME_WON_IMMEDIATE = 0 # Not applicable to Blockdoku typically
R_GAME_LOST_IMMEDIATE = -50.0        # Penalty if a move directly leads to game over (e.g., no more moves possible)
                                     # Note: The GA's main fitness is still game score, these are components of the heuristic.

# --- Genetic Algorithm Parameters ---
GA_POPULATION_SIZE = 20  # Increased for better exploration
GA_NUM_GENERATIONS = 50 # Increased for more evolution
GA_MUTATION_RATE_INITIAL = 0.15
GA_MUTATION_RATE_MIN = 0.02
GA_MUTATION_RATE_MAX = 0.30
GA_MUTATION_STRENGTH_INITIAL = 0.3
GA_MUTATION_STRENGTH_MIN = 0.05
GA_MUTATION_STRENGTH_MAX = 0.5
GA_CROSSOVER_RATE = 0.7
GA_ELITISM_COUNT = 3 # Increased slightly
GA_NUM_GAMES_PER_EVALUATION = 2 # Keep 1 for speed, or increase for robustness (e.g., 3)

# Adaptive Mutation Parameters
GA_ADAPTIVE_MUTATION_STAGNATION_THRESHOLD = 10 # Generations without improvement to trigger increased mutation
GA_ADAPTIVE_MUTATION_RATE_INCREMENT = 0.02
GA_ADAPTIVE_MUTATION_STRENGTH_INCREMENT = 0.02
GA_ADAPTIVE_MUTATION_RATE_DECREMENT = 0.01 # Slight decrease if improving
GA_ADAPTIVE_MUTATION_STRENGTH_DECREMENT = 0.01 # Slight decrease if improving


# Heuristic Definitions for GA Chromosome
# These are the features the GA will learn to weigh.
# The chromosome will be a list of weights corresponding to these.

# Game State Heuristics
H_AGGREGATE_HEIGHT = 0        # Sum of heights of all columns (lower is better)
H_NUM_HOLES = 1               # Number of empty cells blocked from above (lower is better)
H_BUMPINESS = 2               # Sum of height differences between adjacent columns (lower is better)

# Immediate Outcome Heuristics (weights for RL-style rewards if a move is made)
# These allow the GA to learn the importance of immediate rewards vs. board state.
H_IMMEDIATE_BLOCK_PLACED_REWARD = 3
H_IMMEDIATE_CLEAR_REWARD = 4       # Combines line, col, square clear rewards
H_IMMEDIATE_ALMOST_FULL_REWARD = 5
H_IMMEDIATE_GAME_LOST_PENALTY = 6  # If the chosen move directly results in a game over

GA_NUM_HEURISTICS = 7 # Total number of weights in each chromosome

# --- File Paths ---
GA_MODEL_SAVE_DIR = "ga_trained_models"
GA_LOG_FILE = "ga_training_log.csv"
GA_BEST_WEIGHTS_FILE = "best_blockdoku_ga_weights.txt"

# Visualization/Play Settings (for AI playing/training vis) - Used by testing script
VISUALIZE_EVERY_N_EPISODES = 1 # For testing script, visualize first game by default
PLAY_FPS_AI = 10
