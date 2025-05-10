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
def preprocess_human_data(human_data, env): # env is your BlockdokuEnv instance
    """
    Preprocesses human game data by simulating the game using recorded actions
    to reconstruct AI-compatible state-action-reward-next_state-done tuples.

    Args:
        human_data (dict): JSON data of human games, including 'action' and 'reward'.
        env (BlockdokuEnv): The game environment instance for simulation.

    Returns:
        dict: Processed human data with moves as (s, a, r, s', done) tuples
              where s and s' are AI-compatible state dictionaries.
    """
    processed_games_data = [] # Will store lists of processed moves for each game

    for game_idx, game_log in enumerate(human_data.get("games", [])):
        processed_moves_for_this_game = []
        
        # Reset environment to get initial AI-compatible state
        # The 'state' from env.reset() is a dictionary: {'grid': np_array, 'pieces': list_of_keys, 'pieces_spatial': np_array}
        current_ai_state, info = env.reset() 
                                     # info contains 'valid_action_mask', 'available_pieces_keys', etc.

        for move_idx, recorded_move in enumerate(game_log.get("moves", [])):
            human_action_index = recorded_move.get("action")
            # Use "rl_reward_calculated" if present, else "reward"
            human_reward = recorded_move.get("rl_reward_calculated", recorded_move.get("reward", 0.0))

            if human_action_index is None:
                print(f"Warning: Game {game_idx}, Move {move_idx} missing action. Skipping move.")
                continue

            # The state *before* this action is current_ai_state
            state_t_for_buffer = {
                "grid": current_ai_state["grid"].copy(),
                "pieces": current_ai_state["pieces"].copy(), # List of piece keys/IDs
                "pieces_spatial": current_ai_state["pieces_spatial"].copy() # Numpy array for NN
            }

            # --- Optional: Validate human action (highly recommended for debugging) ---
            # valid_action_mask_from_env = info.get("valid_action_mask")
            # if valid_action_mask_from_env is not None and not valid_action_mask_from_env[human_action_index]:
            #     print(f"Warning: Game {game_idx}, Move {move_idx}: Recorded human action {human_action_index} "
            #           f"is NOT valid in the current simulated environment state. Skipping this move.")
            #     # This could happen if the raw log and env simulation diverge.
            #     # You might choose to stop processing this game or try to recover.
            #     # For now, we'll proceed, but it's a point of concern if it happens often.
            #     # Alternatively, if the action is invalid, you cannot env.step() with it.
            #     # So, a better approach might be to break from this game's processing.
            #     # For robust processing, ensure the action is valid or handle appropriately.
            #     # This check is more crucial if your env raises errors on invalid actions.
            #     # If env.step handles invalid actions gracefully (e.g. by doing nothing and returning a penalty),
            #     # then it might be okay. Let's assume env.step can take any action index.

            # Take the recorded human action in the environment
            # env.step() will return the AI-compatible next_state, reward_from_env, done_from_env, next_info
            next_ai_state, reward_from_env, done_from_env, info = env.step(human_action_index)

            state_t_plus_1_for_buffer = {
                "grid": next_ai_state["grid"].copy(),
                "pieces": next_ai_state["pieces"].copy(),
                "pieces_spatial": next_ai_state["pieces_spatial"].copy()
            }
            
            processed_move_tuple = {
                "state": state_t_for_buffer,         # s_t (AI format)
                "action": int(human_action_index),   # a_t (human's action)
                "reward": float(human_reward),       # r_t (human's reward from log)
                "next_state": state_t_plus_1_for_buffer, # s_{t+1} (AI format)
                "done": bool(done_from_env)          # done flag
            }
            processed_moves_for_this_game.append(processed_move_tuple)
            
            # Update current_ai_state for the next iteration
            current_ai_state = next_ai_state
            
            if done_from_env:
                break # End of this game
        
        # Store all processed moves for this game along with its score
        if processed_moves_for_this_game: # Only add if there were valid moves processed
            processed_games_data.append({
                "final_score": game_log.get("final_game_score", game_log.get("final_score", 0)),
                "moves": processed_moves_for_this_game
            })
            
    return {"games": processed_games_data}