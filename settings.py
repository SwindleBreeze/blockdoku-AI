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
R_GAME_LOST_IMMEDIATE = 20.0        # Penalty if a move directly leads to game over (e.g., no more moves possible)
                                     # Note: The GA's main fitness is still game score, these are components of the heuristic.

# --- Genetic Algorithm Parameters ---
# GA_POPULATION_SIZE = 150  # Increased for better exploration
# GA_NUM_GENERATIONS = 300 # Increased for more evolution
# GA_MUTATION_RATE_INITIAL = 0.15
# GA_MUTATION_RATE_MIN = 0.02
# GA_MUTATION_RATE_MAX = 0.30
# GA_MUTATION_STRENGTH_INITIAL = 0.3
# GA_MUTATION_STRENGTH_MIN = 0.05
# GA_MUTATION_STRENGTH_MAX = 0.5
# GA_CROSSOVER_RATE = 0.7
# GA_ELITISM_COUNT = 10 # Increased slightly
# GA_NUM_GAMES_PER_EVALUATION = 3 # Keep 1 for speed, or increase for robustness (e.g., 3)

# RUN 2
# GA_POPULATION_SIZE = 150  # Increased further for broader search
# GA_NUM_GENERATIONS = 300 # Significantly increased for more evolutionary cycles
# GA_MUTATION_RATE_INITIAL = 0.10 # Slightly lower initial rate, adaptive mutation can increase it
# GA_MUTATION_RATE_MIN = 0.01 # Lower minimum for fine-tuning in later stages
# GA_MUTATION_RATE_MAX = 0.35 # Slightly higher max to escape local optima if needed
# GA_MUTATION_STRENGTH_INITIAL = 0.25 # Slightly lower initial strength
# GA_MUTATION_STRENGTH_MIN = 0.02 # Lower minimum strength for finer adjustments
# GA_MUTATION_STRENGTH_MAX = 0.55 # Slightly higher max strength
# GA_CROSSOVER_RATE = 0.75 # Slightly increased to promote mixing of solutions
# GA_ELITISM_COUNT = 10 # Increased proportionally to new population size (approx 7%)
# GA_NUM_GAMES_PER_EVALUATION = 3 # Increased for more robust fitness evaluation

# RUN 3
GA_POPULATION_SIZE = 150  # Increased further for broader search
GA_NUM_GENERATIONS = 140 # Significantly increased for more evolutionary cycles
GA_MUTATION_RATE_INITIAL = 0.10 # Slightly lower initial rate, adaptive mutation can increase it
GA_MUTATION_RATE_MIN = 0.01 # Lower minimum for fine-tuning in later stages
GA_MUTATION_RATE_MAX = 0.30 # Reduced from 0.35 to cap aggressive mutation
GA_MUTATION_STRENGTH_INITIAL = 0.25 # Slightly lower initial strength
GA_MUTATION_STRENGTH_MIN = 0.02 # Lower minimum strength for finer adjustments
GA_MUTATION_STRENGTH_MAX = 0.45 # Reduced from 0.55 to cap aggressive mutation
GA_CROSSOVER_RATE = 0.75 # Slightly increased to promote mixing of solutions
GA_ELITISM_COUNT = 10 # Increased proportionally to new population size (approx 7%)
GA_NUM_GAMES_PER_EVALUATION = 5 # Increased for more robust fitness evaluation

# RUN 4
# GA_POPULATION_SIZE = 150
# GA_NUM_GENERATIONS = 300 # Or more if you want to see if it recovers
# GA_MUTATION_RATE_INITIAL = 0.10
# GA_MUTATION_RATE_MIN = 0.01
# GA_MUTATION_RATE_MAX = 0.25 # Further reduced from 0.30 to prevent overly strong mutations late-game
# GA_MUTATION_STRENGTH_INITIAL = 0.25
# GA_MUTATION_STRENGTH_MIN = 0.02
# GA_MUTATION_STRENGTH_MAX = 0.40 # Further reduced from 0.45
# GA_CROSSOVER_RATE = 0.75
# GA_ELITISM_COUNT = 10
# GA_NUM_GAMES_PER_EVALUATION = 3


# Adaptive Mutation Parameters
# GA_ADAPTIVE_MUTATION_STAGNATION_THRESHOLD = 10 # Generations without improvement to trigger increased mutation
# GA_ADAPTIVE_MUTATION_RATE_INCREMENT = 0.02
# GA_ADAPTIVE_MUTATION_STRENGTH_INCREMENT = 0.02
# GA_ADAPTIVE_MUTATION_RATE_DECREMENT = 0.01 # Slight decrease if improving
# GA_ADAPTIVE_MUTATION_STRENGTH_DECREMENT = 0.01 # Slight decrease if improving

# RUN 3
GA_ADAPTIVE_MUTATION_STAGNATION_THRESHOLD = 15 # Increased from 10 to allow more time before aggressive increase
GA_ADAPTIVE_MUTATION_RATE_INCREMENT = 0.015 # Reduced from 0.02 for a gentler increase
GA_ADAPTIVE_MUTATION_STRENGTH_INCREMENT = 0.015 # Reduced from 0.02 for a gentler increase
GA_ADAPTIVE_MUTATION_RATE_DECREMENT = 0.01 # Slight decrease if improving
GA_ADAPTIVE_MUTATION_STRENGTH_DECREMENT = 0.01 # Slight decrease if improving


# RUN 4
# GA_ADAPTIVE_MUTATION_STAGNATION_THRESHOLD = 20 # Increased from 15, allow more stability before increasing mutation
# GA_ADAPTIVE_MUTATION_RATE_INCREMENT = 0.01  # Reduced from 0.015 for a much gentler increase
# GA_ADAPTIVE_MUTATION_STRENGTH_INCREMENT = 0.01 # Reduced from 0.015 for a much gentler increase
# GA_ADAPTIVE_MUTATION_RATE_DECREMENT = 0.01 # Kept the same, allows reduction if improvements occur
# GA_ADAPTIVE_MUTATION_STRENGTH_DECREMENT = 0.01 # Kept the same


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

GA_BEST_WEIGHTS_FILE = "best_blockdoku_ga_weights.txt"

# Visualization/Play Settings (for AI playing/training vis) - Used by testing script
VISUALIZE_EVERY_N_EPISODES = 1 # For testing script, visualize first game by default
PLAY_FPS_AI = 10
