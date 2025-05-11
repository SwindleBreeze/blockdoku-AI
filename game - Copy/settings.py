# blockdoku/game/settings.py
import pygame

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
GRID_COLOR = (50, 50, 50)
EMPTY_CELL_COLOR = (40, 40, 40)

# --- Grid Dimensions ---
GRID_WIDTH = 9
GRID_HEIGHT = 9
TILE_SIZE = 40
GRID_LINE_WIDTH = 1
BLOCK_BORDER_WIDTH = 1

# --- Screen Dimensions ---
INFO_AREA_WIDTH = 300
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE + INFO_AREA_WIDTH
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE + 80
GRID_START_X = 0
GRID_START_Y = 0

# --- Piece Selection Area ---
PIECE_AREA_X = GRID_WIDTH * TILE_SIZE + 25
PIECE_AREA_Y = 40
PIECE_SPACING = 125

# --- Game Mechanics Settings ---
NUM_PIECES_AVAILABLE = 3 # Also needed by AI for action space
FPS = 60 # For human play and rendering

# --- Piece Definitions (Fundamental to game and AI state) ---
PIECE_DEFINITIONS = {
    'dot': {'shape': [(0, 0)], 'color': (255, 0, 0)},
    'domino_v': {'shape': [(0, 0), (1, 0)], 'color': (0, 255, 0)},
    'domino_h': {'shape': [(0, 0), (0, 1)], 'color': (0, 0, 255)},
    'corner_small': {'shape': [(0, 0), (1, 0), (0, 1)], 'color': (255, 255, 0)},
    'l_shape': {'shape': [(0, 0), (1, 0), (2, 0), (2, 1)], 'color': (0, 255, 255)},
    't_shape': {'shape': [(0,0), (0,1), (0,2), (1,1)], 'color': (255, 0, 255)},
    's_shape': {'shape': [(0,1), (0,2), (1,0), (1,1)], 'color': (255, 165, 0)},
    'z_shape': {'shape': [(0,0), (0,1), (1,1), (1,2)], 'color': (128, 0, 128)},
    'square': {'shape': [(0,0), (0,1), (1,0), (1,1)], 'color': (0, 128, 0)},
    'line_3v': {'shape': [(0,0), (1,0), (2,0)], 'color': (255, 192, 203)},
    'line_3h': {'shape': [(0,0), (0,1), (0,2)], 'color': (165, 42, 42)},
    'u_shape': {'shape': [(0,0), (0,2), (1,0), (1,1), (1,2)], 'color': (75, 0, 130)}
    # Removed 'big_square' for simplicity
    #'big_square': {'shape': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)], 'color': (218, 165, 32)}
}
PIECE_KEYS = list(PIECE_DEFINITIONS.keys())
NUM_PIECE_TYPES = len(PIECE_KEYS) # Needed by AI for state vector
PIECE_KEY_TO_ID = {key: i for i, key in enumerate(PIECE_KEYS)} # Needed by AI

# --- Game Scoring (for display/internal game logic) ---
SCORE_PER_BLOCK_GAME = 1
SCORE_PER_LINE_GAME = 9     # User requested +9 for combo
SCORE_PER_SQUARE_GAME = 9   # User requested +9 for combo

# --- AI related state constants but fundamental to grid observation ---
STATE_GRID_CHANNELS = 1 # Grayscale grid