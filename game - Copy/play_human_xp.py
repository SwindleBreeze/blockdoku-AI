# play_human_xp.py
import pygame
import sys
import json
import os
from game import settings as s_game
from game.grid import Grid
from game.game_state import GameState

# ... (get_grid_coords_from_mouse function remains the same) ...
def get_grid_coords_from_mouse(mouse_x, mouse_y):
    """Converts mouse screen coordinates to grid cell coordinates."""
    if not (s_game.GRID_START_X <= mouse_x < s_game.GRID_START_X + s_game.GRID_WIDTH * s_game.TILE_SIZE and
            s_game.GRID_START_Y <= mouse_y < s_game.GRID_START_Y + s_game.GRID_HEIGHT * s_game.TILE_SIZE):
        return None

    grid_c = (mouse_x - s_game.GRID_START_X) // s_game.TILE_SIZE
    grid_r = (mouse_y - s_game.GRID_START_Y) // s_game.TILE_SIZE
    return grid_r, grid_c

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT))
    pygame.display.set_caption("Blockdoku")
    clock = pygame.time.Clock()

    # Define the output directory and filename with OS path handling

    #output_dir = "/Users/aljazjustin/soal-programi/MAGI/UI/blockdoku-AI/human_games" # Consider making this path more relative or configurable
    output_dir = os.path.join(os.path.dirname(__file__), "human_games") # Relative to the script's location
    # output_dir = os.path.join(os.getcwd(), "human_games") # Current working directory
    output_filename = "recorded_human_games.json"
    output_filepath = os.path.join(output_dir, output_filename)

    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        # print(f"Directory '{output_dir}' ensured or already exists.") # Optional print
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}. Cannot save games.")
        return # Exit if we can't ensure directory

    # --- MODIFICATION START: Load existing game data ---
    all_games_data_container = {"games": []} # Initialize a default structure
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r') as f:
                all_games_data_container = json.load(f)
                if not isinstance(all_games_data_container, dict) or "games" not in all_games_data_container:
                    print(f"Warning: File {output_filepath} has unexpected format. Starting with empty game list.")
                    all_games_data_container = {"games": []}
                elif not isinstance(all_games_data_container["games"], list):
                    print(f"Warning: 'games' in {output_filepath} is not a list. Starting with empty game list.")
                    all_games_data_container["games"] = []

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {output_filepath}. Starting with empty game list.")
            all_games_data_container = {"games": []} # Reset if file is corrupt
        except Exception as e:
            print(f"Error loading existing game data: {e}. Starting with empty game list.")
            all_games_data_container = {"games": []}
    
    # 'games_list' will be the list we append to.
    games_list_to_append_to = all_games_data_container.get("games", [])
    if not isinstance(games_list_to_append_to, list): # Ensure it's a list
        print("Warning: Corrected 'games' field to be a list.")
        games_list_to_append_to = []

    # --- MODIFICATION END ---

    grid = Grid()
    game = GameState(grid)

    current_game_moves = []
    selected_piece_original_index = -1
    selected_piece = None
    mouse_offset_x = 0
    mouse_offset_y = 0

    running = True
    game_played_this_session = False # Flag to check if a game was actually played

    while running:
        if not game_played_this_session and not game.game_over : # Example condition to start game logic
             game_played_this_session = True # Mark that we're interacting with a game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game.game_over:
                # Game over logic for the current game session
                # We'll handle saving when the main loop 'running' becomes false,
                # or you can add a "Play Again?" prompt here.
                # For simplicity, this example saves when pygame quits.
                continue

            # ... (rest of your MOUSEBUTTONDOWN and MOUSEBUTTONUP event handling logic) ...
            # Ensure current_game_moves is populated correctly within this loop
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for idx, piece in enumerate(game.current_pieces):
                        if piece.get_bounds().collidepoint(event.pos):
                            selected_piece = piece
                            selected_piece.is_dragging = True
                            try:
                                selected_piece_original_index = game.current_pieces.index(selected_piece)
                            except ValueError:
                                selected_piece_original_index = -1
                            mouse_offset_x = event.pos[0] - selected_piece.screen_pos[0]
                            mouse_offset_y = event.pos[1] - selected_piece.screen_pos[1]
                            break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selected_piece is not None:
                    grid_state_for_log = [row[:] for row in grid.grid_data]
                    pieces_identifiers_for_log = []
                    pieces_spatial_for_log = [] # You might not need to log raw spatial, if your AI converts from ID

                    for p_idx, p_obj in enumerate(game.current_pieces):
                        pieces_identifiers_for_log.append(p_obj.id if hasattr(p_obj, 'id') else f"piece_type_{p_idx}")
                        # pieces_spatial_for_log.append(p_obj.shape if hasattr(p_obj, 'shape') else []) # Example

                    # It's good practice to use the keys that your `preprocess_human_data` expects for the raw log
                    state_for_log = {
                        "grid_raw": grid_state_for_log, # Match expected key if `preprocess_human_data` uses this
                        "available_pieces_shape_keys": pieces_identifiers_for_log # Match expected key
                        # "pieces_spatial": pieces_spatial_for_log # Optional, if logged
                    }

                    grid_coords = get_grid_coords_from_mouse(
                        event.pos[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                        event.pos[1] - mouse_offset_y + s_game.TILE_SIZE // 2
                    )
                    
                    placed = False
                    if grid_coords:
                        target_r, target_c = grid_coords
                        action_for_log = -1
                        if 0 <= selected_piece_original_index < len(game.current_pieces):
                            action_for_log = (selected_piece_original_index * (s_game.GRID_WIDTH * s_game.GRID_HEIGHT) +
                                              target_r * s_game.GRID_WIDTH + target_c)

                        placement_results = game.attempt_placement(selected_piece, target_r, target_c)
                        success_place = placement_results[0]
                        reward_for_log = 0.0
                        if len(placement_results) >= 5:
                            reward_for_log = float(placement_results[4]) # This is score_increment
                        
                        if success_place:
                            placed = True
                            current_game_moves.append({
                                # Using more descriptive keys that match example JSON structure
                                "state_raw_log": state_for_log,
                                "action": action_for_log,
                                "rl_reward_calculated": reward_for_log # Or "reward"
                            })
                    
                    if not placed:
                        selected_piece.reset_position()
                    selected_piece.is_dragging = False
                    selected_piece = None
                    selected_piece_original_index = -1


        screen.fill(s_game.BLACK)
        grid.draw(screen)
        game.draw_score(screen)
        game.draw_available_pieces(screen)
        if selected_piece and selected_piece.is_dragging: # Ghost piece logic
            grid_coords_ghost = get_grid_coords_from_mouse(
                pygame.mouse.get_pos()[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                pygame.mouse.get_pos()[1] - mouse_offset_y + s_game.TILE_SIZE // 2
            )
            if grid_coords_ghost:
                target_r_g, target_c_g = grid_coords_ghost
                if grid.can_place_piece(selected_piece, target_r_g, target_c_g):
                    original_screen_pos = selected_piece.screen_pos
                    selected_piece.screen_pos = (
                        s_game.GRID_START_X + target_c_g * s_game.TILE_SIZE,
                        s_game.GRID_START_Y + target_r_g * s_game.TILE_SIZE
                    )
                    selected_piece.draw(screen, ghost=True)
                    selected_piece.screen_pos = original_screen_pos
        if selected_piece is not None and selected_piece.is_dragging:
            selected_piece.draw(screen)
        game.draw_game_over(screen)
        pygame.display.flip()
        clock.tick(s_game.FPS)

    # After the game loop (when running is False)
    # --- MODIFICATION START: Append current game data and save all ---
    if game_played_this_session and current_game_moves: # Only save if moves were made in this session
        final_score = game.score # Get final score of the game just played
        game_session_log = {
            "final_score": final_score, # Or "final_score" to match your JSON
            "moves": current_game_moves
        }
        games_list_to_append_to.append(game_session_log) # Append the new game to the loaded list

        # Update the main container
        all_games_data_container["games"] = games_list_to_append_to

        print(f"\n--- Saving Game Log to {output_filepath} ---")
        try:
            with open(output_filepath, 'w') as f: # Still use 'w' to write the whole updated list
                json.dump(all_games_data_container, f, indent=2)
            print(f"Successfully saved game log ({len(games_list_to_append_to)} total games) to {output_filepath}")
        except TypeError as e:
            print(f"Error serializing to JSON: {e}")
        except IOError as e:
            print(f"Error writing to file {output_filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during saving: {e}")
    elif game_played_this_session:
        print("No moves made in this session, game data not saved.")
    else:
        print("No game was played this session, nothing to save.")
    # --- MODIFICATION END ---

    print("--- End of Logging ---")
    pygame.font.quit()
    pygame.quit()

if __name__ == '__main__':
    main()