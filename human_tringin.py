# play_bd.py
import pygame
import sys
import json # Added for logging output
import os 
# Explicitly import game settings for game visuals and mechanics
from game import settings as s_game 
from game.grid import Grid
from game.game_state import GameState
# Piece is used but its settings come from s_game indirectly through PIECE_DEFINITIONS
# from game.piece import Piece 

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
    pygame.font.init() # Initialize font module
    screen = pygame.display.set_mode((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT))
    pygame.display.set_caption("Blockdoku")
    clock = pygame.time.Clock()

    grid = Grid()
    game = GameState(grid)

    # Logging variables
    all_games_data = []
    current_game_moves = []
    selected_piece_original_index = -1 # Index of the piece in game.current_pieces when selected

    selected_piece = None # Explicitly type hint if possible: Optional[Piece] = None
    mouse_offset_x = 0
    mouse_offset_y = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game.game_over: 
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    for idx, piece in enumerate(game.current_pieces):
                        if piece.get_bounds().collidepoint(event.pos):
                            selected_piece = piece
                            selected_piece.is_dragging = True
                            try:
                                # Store index from the perspective of available pieces
                                selected_piece_original_index = game.current_pieces.index(selected_piece)
                            except ValueError:
                                selected_piece_original_index = -1 # Should not happen
                            mouse_offset_x = event.pos[0] - selected_piece.screen_pos[0]
                            mouse_offset_y = event.pos[1] - selected_piece.screen_pos[1]
                            break 

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selected_piece is not None:
                    # Capture state BEFORE attempting placement for logging
                    grid_state_for_log = [row[:] for row in grid.grid_data] # <--- Corrected line 70
                    
                    pieces_identifiers_for_log = []
                    pieces_spatial_for_log = []
                    for p_idx, p_obj in enumerate(game.current_pieces):
                        try:
                            # Assuming Piece object has an 'id' or 'type' attribute
                            pieces_identifiers_for_log.append(p_obj.id if hasattr(p_obj, 'id') else f"piece_type_{p_idx}")
                        except AttributeError:
                            pieces_identifiers_for_log.append(f"unknown_piece_{p_idx}")
                        try:
                            # Assuming Piece object has a 'shape' attribute (list of lists)
                            pieces_spatial_for_log.append(p_obj.shape if hasattr(p_obj, 'shape') else [])
                        except AttributeError:
                            pieces_spatial_for_log.append([])

                    state_for_log = {
                        "grid": grid_state_for_log,
                        "pieces": pieces_identifiers_for_log,
                        "pieces_spatial": pieces_spatial_for_log
                    }

                    grid_coords = get_grid_coords_from_mouse(
                        event.pos[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                        event.pos[1] - mouse_offset_y + s_game.TILE_SIZE // 2
                    )
                    
                    placed = False
                    if grid_coords:
                        target_r, target_c = grid_coords
                        
                        action_for_log = -1 # Default invalid action
                        if 0 <= selected_piece_original_index < len(game.current_pieces): # Check validity
                            action_for_log = (selected_piece_original_index * (s_game.GRID_WIDTH * s_game.GRID_HEIGHT) +
                                              target_r * s_game.GRID_WIDTH + target_c)

                        # attempt_placement returns success, r_cleared, c_cleared, sq_cleared, score_increment (and potentially more)
                        placement_results = game.attempt_placement(selected_piece, target_r, target_c)
                        success_place = placement_results[0]
                        
                        reward_for_log = 0.0
                        if len(placement_results) >= 5: # Assuming score_increment is the 5th element
                            reward_for_log = float(placement_results[4])
                        
                        if success_place:
                            placed = True
                            current_game_moves.append({
                                "state": state_for_log,
                                "action": action_for_log,
                                "reward": reward_for_log
                            })
                    
                    if not placed:
                        selected_piece.reset_position()

                    selected_piece.is_dragging = False
                    selected_piece = None
                    selected_piece_original_index = -1 # Reset index

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selected_piece is not None:
                    # Capture state BEFORE attempting placement for logging
                    grid_state_for_log = [row[:] for row in grid._grid] # Changed grid.grid to grid._grid
                    
                    pieces_identifiers_for_log = []

        screen.fill(s_game.BLACK)
        grid.draw(screen)
        game.draw_score(screen)
        game.draw_available_pieces(screen)

        # Ghost piece logic (optional, can be enhanced)
        if selected_piece and selected_piece.is_dragging:
             grid_coords_ghost = get_grid_coords_from_mouse(
                 pygame.mouse.get_pos()[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                 pygame.mouse.get_pos()[1] - mouse_offset_y + s_game.TILE_SIZE // 2
             )
             if grid_coords_ghost:
                 target_r_g, target_c_g = grid_coords_ghost
                 if grid.can_place_piece(selected_piece, target_r_g, target_c_g):
                    original_screen_pos = selected_piece.screen_pos # Store original
                    selected_piece.screen_pos = (
                        s_game.GRID_START_X + target_c_g * s_game.TILE_SIZE,
                        s_game.GRID_START_Y + target_r_g * s_game.TILE_SIZE
                    )
                    selected_piece.draw(screen, ghost=True) # Assuming Piece.draw handles ghost
                    selected_piece.screen_pos = original_screen_pos # Restore

        if selected_piece is not None and selected_piece.is_dragging:
            selected_piece.draw(screen) # Draw the actual dragged piece

        game.draw_game_over(screen)

        pygame.display.flip()
        clock.tick(s_game.FPS)

    # After the game loop, prepare the log for the completed game
    # After the game loop, prepare the log for the completed game
    final_score = game.score
    game_session_log = {
        "final_score": final_score,
        "moves": current_game_moves
    }
    all_games_data.append(game_session_log)

    # Define the output directory and filename
    output_dir = "/Users/aljazjustin/soal-programi/MAGI/UI/blockdoku-AI/human_games"
    output_filename = "recorded_human_games.json"
    output_filepath = os.path.join(output_dir, output_filename)

    # Create the directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory '{output_dir}' ensured or already exists.")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        # Optionally, handle this error more gracefully, e.g., by not attempting to save

    # Save the collected data to a JSON file
    print(f"\n--- Saving Game Log to {output_filepath} ---")
    try:
        with open(output_filepath, 'w') as f:
            json.dump({"games": all_games_data}, f, indent=2)
        print(f"Successfully saved game log to {output_filepath}")
    except TypeError as e:
        print(f"Error serializing to JSON: {e}")
        print("Ensure piece shapes and IDs are JSON serializable (e.g., lists, strings, numbers).")
    except IOError as e:
        print(f"Error writing to file {output_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")
    print("--- End of Logging ---")

    pygame.font.quit()
    pygame.quit()

if __name__ == '__main__':
    main()