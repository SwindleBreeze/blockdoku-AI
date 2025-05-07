# utils.py
import settings as s_ai # AI settings which imports necessary game dimensions
import numpy as np
# game_state module is needed for type hinting, and get_valid_action_mask operates on GameState
from game.game_state import GameState 
# piece module might be needed if GameState.current_pieces directly exposes Piece objects
# from game.piece import Piece 


def decode_action(action_index):
    """ Converts a flat action index to (piece_index, grid_r, grid_c). """
    if not (0 <= action_index < s_ai.ACTION_SPACE_SIZE):
        raise ValueError(f"Action index {action_index} out of bounds for AI action space {s_ai.ACTION_SPACE_SIZE}.")

    actions_per_row_col_grid = s_ai.GRID_WIDTH * s_ai.GRID_HEIGHT
    piece_index = action_index // actions_per_row_col_grid
    remainder = action_index % actions_per_row_col_grid
    grid_r = remainder // s_ai.GRID_WIDTH
    grid_c = remainder % s_ai.GRID_WIDTH

    return piece_index, grid_r, grid_c

def encode_action(piece_index, grid_r, grid_c):
    """ Converts (piece_index, grid_r, grid_c) to a flat action index. """
    if not (0 <= piece_index < s_ai.NUM_PIECES_AVAILABLE and
            0 <= grid_r < s_ai.GRID_HEIGHT and
            0 <= grid_c < s_ai.GRID_WIDTH):
         raise ValueError(f"Invalid action components for encoding: p_idx={piece_index}, r={grid_r}, c={grid_c}")

    return piece_index * (s_ai.GRID_WIDTH * s_ai.GRID_HEIGHT) + grid_r * s_ai.GRID_WIDTH + grid_c

def get_valid_action_mask(game_state: GameState) -> np.ndarray: # Type hint for clarity
    """ Returns a boolean mask (size ACTION_SPACE_SIZE) indicating valid actions. """
    mask = np.zeros(s_ai.ACTION_SPACE_SIZE, dtype=bool)
    current_pieces = game_state.current_pieces 
    grid = game_state.grid # This grid object uses game.settings for its dimensions

    for idx, piece in enumerate(current_pieces): # idx will be 0, 1, 2
        # grid.height and grid.width come from s_game.GRID_HEIGHT/WIDTH via grid's init
        for r in range(grid.height): 
            for c in range(grid.width):
                if grid.can_place_piece(piece, r, c): # piece is a Piece object
                    try:
                        action_idx = encode_action(idx, r, c) # Uses s_ai dimensions
                        if 0 <= action_idx < s_ai.ACTION_SPACE_SIZE:
                             mask[action_idx] = True
                    except ValueError:
                        # This should not happen if loops and NUM_PIECES_AVAILABLE are correct
                        print(f"Error encoding action for mask: p_idx={idx}, r={r}, c={c}")
    return mask