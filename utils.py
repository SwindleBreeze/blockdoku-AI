# utils.py
import settings as s_ai # AI settings which imports necessary game dimensions
import numpy as np
# game_state module is needed for type hinting, and get_valid_action_mask operates on GameState
from game.game_state import GameState 
# piece module might be needed if GameState.current_pieces directly exposes Piece objects
# from game.piece import Piece 
import game.settings as s_game # Game settings for grid dimensions

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

def get_piece_spatial_representation(piece_keys):
    """Convert piece keys to spatial 4x4 grid representations"""
    from game.settings import PIECE_DEFINITIONS
    
    piece_grids = np.zeros((3, 4, 4), dtype=np.float32)  # 3 pieces, each on a 4x4 grid
    
    for i, piece_key in enumerate(piece_keys):
        if piece_key is None or piece_key not in PIECE_DEFINITIONS:
            continue
            
        piece_shape = PIECE_DEFINITIONS[piece_key]['shape']
        
        # Place piece on the 4x4 grid - centered at (1,1)
        for y, x in piece_shape:
            if 0 <= y < 4 and 0 <= x < 4:  # Ensure within bounds
                piece_grids[i, y, x] = 1.0
    
    return piece_grids
def preprocess_human_data(human_data, env):
    """
    Preprocesses human game data that only has reward information
    by simulating the game to reconstruct state-action pairs.
    
    Args:
        human_data (dict): JSON data of human games
        env (BlockdokuEnv): Environment to simulate moves
        
    Returns:
        dict: Processed human data with full state representations
    """
    processed_games = []
    
    for game_idx, game in enumerate(human_data.get("games", [])):
        processed_moves = []
        
        # Reset environment to get initial state
        state, info = env.reset()
        
        for move_idx, move in enumerate(game.get("moves", [])):
            # We need to generate valid actions for this state
            valid_actions = np.where(info.get("valid_action_mask", None))[0]
            
            if len(valid_actions) == 0:
                print(f"Warning: No valid actions for game {game_idx}, move {move_idx}")
                continue
                
            # Select a valid action based on highest expected reward
            # This is a simplification - in reality we would want to match with the actual human action
            action = valid_actions[0]
            
            # Record the state and action
            processed_move = {
                "state": {
                    "grid": state["grid"].copy(),
                    "pieces": state["pieces"].copy(),
                    "pieces_spatial": state["pieces_spatial"].copy()
                },
                "action": int(action),
                "reward": float(move.get("reward", 0.0))
            }
            processed_moves.append(processed_move)
            
            # Take the action in the environment to advance to the next state
            next_state, reward, done, info = env.step(action)
            
            # Update the state for next move
            state = next_state
            
            if done:
                break
        
        processed_games.append({
            "final_score": game.get("final_score", 0),
            "moves": processed_moves
        })
    
    return {"games": processed_games}