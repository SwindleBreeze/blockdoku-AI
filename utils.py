# utils.py
import settings as s
import numpy as np

def decode_action(action_index):
    """ Converts a flat action index (0-242) to (piece_index, grid_r, grid_c). """
    if not (0 <= action_index < s.ACTION_SPACE_SIZE):
        raise ValueError(f"Action index {action_index} out of bounds.")

    piece_index = action_index // (s.GRID_WIDTH * s.GRID_HEIGHT)
    remainder = action_index % (s.GRID_WIDTH * s.GRID_HEIGHT)
    grid_r = remainder // s.GRID_WIDTH
    grid_c = remainder % s.GRID_WIDTH

    return piece_index, grid_r, grid_c

def encode_action(piece_index, grid_r, grid_c):
    """ Converts (piece_index, grid_r, grid_c) to a flat action index. """
    if not (0 <= piece_index < s.NUM_PIECES_AVAILABLE and
            0 <= grid_r < s.GRID_HEIGHT and
            0 <= grid_c < s.GRID_WIDTH):
         raise ValueError("Invalid action components for encoding.")

    return piece_index * (s.GRID_WIDTH * s.GRID_HEIGHT) + grid_r * s.GRID_WIDTH + grid_c

# --- Action Masking Function (Optional but Recommended) ---
def get_valid_action_mask(game_state):
    """ Returns a boolean mask (size ACTION_SPACE_SIZE) indicating valid actions. """
    mask = np.zeros(s.ACTION_SPACE_SIZE, dtype=bool)
    current_pieces = game_state.current_pieces # Get Piece objects
    grid = game_state.grid

    for idx, piece in enumerate(current_pieces):
        for r in range(s.GRID_HEIGHT):
            for c in range(s.GRID_WIDTH):
                if grid.can_place_piece(piece, r, c):
                    action_idx = encode_action(idx, r, c)
                    if 0 <= action_idx < s.ACTION_SPACE_SIZE:
                         mask[action_idx] = True
    return mask