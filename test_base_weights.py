# test_best_weights.py
import pygame
import numpy as np
import os
import sys
import time
import copy
from tqdm import tqdm

# Game and Utils Imports
try:
    from game.game_state import GameState
    from game.grid import Grid
    from game import settings as s_game # Game core settings for display
    import utils
    import settings as s_ai # AI settings for paths and heuristic definitions
    # Import heuristic calculation functions from genetic_solver
    from genetic_alg import (
        calculate_aggregate_height, calculate_num_holes, calculate_bumpiness,
        evaluate_potential_move # evaluate_potential_move is key
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure all required files (game modules, utils.py, settings.py, genetic_solver.py) are accessible.")
    sys.exit(1)

def load_best_weights(filepath=None) -> list[float] | None:
    """Loads the best weights from the specified file."""
    if filepath is None:
        filepath = os.path.join(s_ai.GA_MODEL_SAVE_DIR, s_ai.GA_BEST_WEIGHTS_FILE)
    
    if not os.path.exists(filepath):
        print(f"Error: Weights file not found at {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            weights_str = f.read()
            weights = [float(w.strip()) for w in weights_str.split(',')]
        if len(weights) == s_ai.GA_NUM_HEURISTICS:
            print(f"Successfully loaded {len(weights)} weights from {filepath}")
            return weights
        else:
            print(f"Error: Number of weights in file ({len(weights)}) does not match GA_NUM_HEURISTICS ({s_ai.GA_NUM_HEURISTICS}).")
            return None
    except Exception as e:
        print(f"Error reading weights file: {e}")
        return None

def play_game_with_weights(weights: list[float], render: bool = False, fps: int = s_ai.PLAY_FPS_AI) -> int:
    """
    Plays a single game of Blockdoku using the provided heuristic weights.
    Returns the final game score.
    """
    game_grid = Grid()
    game_state = GameState(game_grid) # GameState handles pygame.font.init

    if render:
        screen = pygame.display.set_mode((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT))
        pygame.display.set_caption("Blockdoku - GA Agent Test")
        clock = pygame.time.Clock()

    while not game_state.game_over:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return game_state.score # Or sys.exit()
            screen.fill(s_game.BLACK)
            game_grid.draw(screen)
            game_state.draw_score(screen)
            game_state.draw_available_pieces(screen) # Draws pieces at their origin

        best_move_details = None
        valid_actions_mask = utils.get_valid_action_mask(game_state)
        possible_moves_exist_for_turn = False

        if not game_state.current_pieces:
            game_state._generate_new_pieces()
            if game_state.game_over: break

        for action_idx in range(s_ai.ACTION_SPACE_SIZE):
            if valid_actions_mask[action_idx]:
                possible_moves_exist_for_turn = True
                decoded_p_idx, decoded_r, decoded_c = utils.decode_action(action_idx)

                if decoded_p_idx >= len(game_state.current_pieces): continue
                current_piece = game_state.current_pieces[decoded_p_idx]

                sim_grid_post_placement_pre_clear = copy.deepcopy(game_grid)
                sim_grid_post_placement_pre_clear.place_piece(current_piece, decoded_r, decoded_c)
                
                move_outcome_details = {}
                move_outcome_details['block_placed_reward'] = s_ai.R_PLACED_BLOCK_IMMEDIATE
                piece_cells_on_grid = {(decoded_r + ro, decoded_c + co) for ro, co in current_piece.relative_cells}
                af_count, _ = sim_grid_post_placement_pre_clear.get_almost_full_regions_info(piece_cells_on_grid)
                move_outcome_details['almost_full_reward'] = af_count * s_ai.R_ALMOST_FULL_IMMEDIATE
                temp_grid_for_clear_counting = copy.deepcopy(sim_grid_post_placement_pre_clear)
                cleared_r, cleared_c, cleared_sq = temp_grid_for_clear_counting.clear_lines_and_squares()
                total_clears = cleared_r + cleared_c + cleared_sq
                move_outcome_details['clear_reward'] = total_clears * s_ai.R_CLEARED_LINE_COL_IMMEDIATE
                
                # For testing, game_lost_penalty is implicitly handled by low scores if the game ends.
                # The heuristic is primarily for guiding choices, not for exact reward shaping during test.
                move_outcome_details['game_lost_penalty'] = 0 # Not actively used in test eval this way

                current_move_eval_score = evaluate_potential_move(
                    sim_grid_post_placement_pre_clear,
                    weights,
                    move_outcome_details
                )

                if best_move_details is None or current_move_eval_score > best_move_details[4]:
                    best_move_details = (decoded_p_idx, current_piece, decoded_r, decoded_c, current_move_eval_score, action_idx)
        
        if not possible_moves_exist_for_turn or best_move_details is None:
            game_state.game_over = True
            break

        actual_piece_to_place = game_state.current_pieces[best_move_details[0]]
        target_r_actual, target_c_actual = best_move_details[2], best_move_details[3]
        placement_results = game_state.attempt_placement(actual_piece_to_place, target_r_actual, target_c_actual)

        if not placement_results[0]:
            game_state.game_over = True
            break
        
        if render:
            if game_state.game_over: # Draw game over screen before exiting loop
                 game_state.draw_game_over(screen)
            pygame.display.flip()
            clock.tick(fps)
            # time.sleep(0.1) # Optional delay to make it easier to watch

    if render: # Final game over screen display if loop exited due to game over
        if game_state.game_over and not pygame.display.get_init(): # If quit event closed display
            pass # Pygame already quit
        elif game_state.game_over:
            screen.fill(s_game.BLACK) # Redraw background
            game_grid.draw(screen)
            game_state.draw_score(screen)
            # game_state.draw_available_pieces(screen) # Pieces might be empty
            game_state.draw_game_over(screen)
            pygame.display.flip()
            # Keep game over screen for a bit
            start_time = time.time()
            while time.time() - start_time < 3: # Show for 3 seconds
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return game_state.score
                 if not pygame.display.get_init(): break # If display closed by event
                 clock.tick(10)


    return game_state.score

def main_test_and_validate(num_test_games=100, visualize_first_n_games=1):
    """
    Loads best weights, runs multiple test games, and prints statistics.
    Optionally visualizes the first few games.
    """
    print("--- Testing and Validating Best GA Weights ---")
    
    # Pygame init needed for GameState, even if not rendering all games
    pygame.init() 
    pygame.font.init()

    best_weights = load_best_weights()
    if best_weights is None:
        print("Could not proceed with testing.")
        if pygame.get_init(): pygame.quit()
        return

    print(f"Testing with weights: {[f'{w:.3f}' for w in best_weights]}")

    scores = []
    for i in tqdm(range(num_test_games), desc="Running Test Games"):
        render_this_game = i < visualize_first_n_games
        score = play_game_with_weights(best_weights, render=render_this_game)
        scores.append(score)
        if render_this_game and i == visualize_first_n_games - 1 and visualize_first_n_games < num_test_games:
            print(f"Finished visualizing {visualize_first_n_games} game(s). Running remaining tests headlessly...")


    if scores:
        avg_score = np.mean(scores)
        std_dev_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        median_score = np.median(scores)

        print("\n--- Test Results ---")
        print(f"Number of test games: {len(scores)}")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Median Score: {median_score:.2f}")
        print(f"Standard Deviation of Score: {std_dev_score:.2f}")
        print(f"Min Score: {min_score}")
        print(f"Max Score: {max_score}")
    else:
        print("No games were played or no scores recorded.")

    if pygame.get_init():
        pygame.quit()

if __name__ == "__main__":
    num_games_to_test = 100  # How many games to run for statistics
    num_games_to_visualize = 1 # How many of the first games to render

    # Example: python test_best_weights.py 50 2  (to test 50 games, visualize first 2)
    if len(sys.argv) > 1:
        try:
            num_games_to_test = int(sys.argv[1])
            if len(sys.argv) > 2:
                num_games_to_visualize = int(sys.argv[2])
        except ValueError:
            print("Usage: python test_best_weights.py [num_test_games] [num_visualize_games]")
            print("Using default values.")

    main_test_and_validate(num_test_games=num_games_to_test, visualize_first_n_games=num_games_to_visualize)
