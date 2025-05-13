# genetic_solver.py
import random
import copy
import sys
import os
import csv
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count # Added
from functools import partial # Added
from datetime import datetime 

# --- Game Imports ---
try:
    from game.game_state import GameState
    from game.grid import Grid
except ImportError as e:
    print(f"Error importing game modules: {e}")
    sys.exit(1)

# --- AI Utils and Settings Imports ---
try:
    import utils
    import settings as s_ai # AI settings file
except ImportError as e:
    print(f"Error importing 'utils.py' or AI 'settings.py': {e}")
    sys.exit(1)

# --- Global GA Parameters (loaded from s_ai) ---
# These will be assigned in run_ga to allow for easier modification via settings
current_mutation_rate = s_ai.GA_MUTATION_RATE_INITIAL
current_mutation_strength = s_ai.GA_MUTATION_STRENGTH_INITIAL

# It's good practice to ensure functions run by worker processes can initialize necessary libraries
# if they rely on global state from the main process that isn't inherited.
# For Pygame, if GameState or Grid directly use pygame.font without checking/re-initializing:

def worker_init_pygame():
    """Initializes Pygame modules for a worker process if necessary."""
    try:
        import pygame
        pygame.init() # Basic pygame init
        pygame.font.init() # Specifically for font loading
        # print("Pygame (and font) initialized in worker.") # For debugging
    except Exception as e:
        # print(f"Worker Pygame init warning: {e}")
        pass # Continue if non-critical or handled within GameState


# --- Helper Functions for Calculating Heuristics ---
def calculate_aggregate_height(grid_obj: Grid) -> int:
    heights = [0] * grid_obj.width
    for c in range(grid_obj.width):
        for r_idx in range(grid_obj.height):
            if grid_obj.grid_data[r_idx][c] is not None:
                heights[c] = grid_obj.height - r_idx
                break
    return sum(heights)

def calculate_num_holes(grid_obj: Grid) -> int:
    holes = 0
    for c in range(grid_obj.width):
        column_has_block_above = False
        for r_idx in range(grid_obj.height):
            if grid_obj.grid_data[r_idx][c] is not None:
                column_has_block_above = True
            elif column_has_block_above and grid_obj.grid_data[r_idx][c] is None:
                holes += 1
    return holes

def calculate_bumpiness(grid_obj: Grid) -> int:
    heights = [0] * grid_obj.width
    for c in range(grid_obj.width):
        for r_idx in range(grid_obj.height):
            if grid_obj.grid_data[r_idx][c] is not None:
                heights[c] = grid_obj.height - r_idx
                break
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])
    return bumpiness

# --- Chromosome (Individual) Definition ---
def create_individual() -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(s_ai.GA_NUM_HEURISTICS)]

# --- Evaluation Function (Scores a single potential move) ---
def evaluate_potential_move(
    grid_state_after_placement_before_clears: Grid, # Grid state for board heuristics
    weights: list[float],
    move_outcome_details: dict # Contains immediate rewards and clear counts
) -> float:
    """
    Calculates a score for a potential move based on the chromosome's weights.
    Combines board state heuristics and immediate outcome rewards.
    """
    heuristic_score = 0.0
    
    # 1. Board State Heuristics (evaluated on the grid *after* piece placement, *before* clears)
    agg_height = calculate_aggregate_height(grid_state_after_placement_before_clears)
    num_holes = calculate_num_holes(grid_state_after_placement_before_clears)
    bumpiness_val = calculate_bumpiness(grid_state_after_placement_before_clears)

    heuristic_score += weights[s_ai.H_AGGREGATE_HEIGHT] * agg_height
    heuristic_score += weights[s_ai.H_NUM_HOLES] * num_holes
    heuristic_score += weights[s_ai.H_BUMPINESS] * bumpiness_val

    # 2. Immediate Outcome Heuristics (based on what the move achieves)
    # These weights determine how much the GA values these immediate rewards
    # relative to the board state heuristics.
    heuristic_score += weights[s_ai.H_IMMEDIATE_BLOCK_PLACED_REWARD] * move_outcome_details.get('block_placed_reward', 0)
    heuristic_score += weights[s_ai.H_IMMEDIATE_CLEAR_REWARD] * move_outcome_details.get('clear_reward', 0)
    heuristic_score += weights[s_ai.H_IMMEDIATE_ALMOST_FULL_REWARD] * move_outcome_details.get('almost_full_reward', 0)
    heuristic_score += weights[s_ai.H_IMMEDIATE_GAME_LOST_PENALTY] * move_outcome_details.get('game_lost_penalty', 0)
    
    return heuristic_score

# --- Fitness Function (Plays games and returns average score) ---
def calculate_fitness(individual_weights: list[float], num_games: int = s_ai.GA_NUM_GAMES_PER_EVALUATION) -> float:
    total_score_across_all_games = 0.0

    for _ in range(num_games):
        game_grid = Grid()
        game_state = GameState(game_grid)

        while not game_state.game_over:
            best_move_details = None
            valid_actions_mask = utils.get_valid_action_mask(game_state)
            possible_moves_exist_for_turn = False

            if not game_state.current_pieces: # Should be handled by _generate_new_pieces
                game_state._generate_new_pieces()
                if game_state.game_over: break # Game over immediately after new pieces

            for action_idx in range(s_ai.ACTION_SPACE_SIZE):
                if valid_actions_mask[action_idx]:
                    possible_moves_exist_for_turn = True
                    decoded_p_idx, decoded_r, decoded_c = utils.decode_action(action_idx)

                    if decoded_p_idx >= len(game_state.current_pieces): continue
                    current_piece = game_state.current_pieces[decoded_p_idx]

                    # print( f"Evaluating action {action_idx}: Piece {decoded_p_idx}, Position ({decoded_r}, {decoded_c})")
                    # sim_grid_post_placement_pre_clear = copy.deepcopy(game_grid)
                    cleared_r, cleared_c, cleared_sq,af_count,sim_game_over,sim_grid =game_state.simulate_attempt_placement(current_piece, decoded_r, decoded_c)
                    
                    # print( cleared_r, cleared_c, cleared_sq,af_count)
                    # Calculate immediate rewards and clear counts for this potential move
                    move_outcome_details = {}
                    move_outcome_details['block_placed_reward'] = s_ai.R_PLACED_BLOCK_IMMEDIATE

                    # piece_cells_on_grid = {(decoded_r + ro, decoded_c + co) for ro, co in current_piece.relative_cells}
                    # af_count, _ = sim_game_state.grid.get_almost_full_regions_info(piece_cells_on_grid)
                    move_outcome_details['almost_full_reward'] = af_count * s_ai.R_ALMOST_FULL_IMMEDIATE
                    # print(f"Almost full count: {af_count} for piece {current_piece}")
                    # temp_grid_for_clear_counting = copy.deepcopy(sim_grid_post_placement_pre_clear)
                    # cleared_r, cleared_c, cleared_sq = sim_game_state.grid.clear_lines_and_squares()
                    total_clears = cleared_r + cleared_c + cleared_sq
                    move_outcome_details['clear_reward'] = total_clears * s_ai.R_CLEARED_LINE_COL_IMMEDIATE # Assuming same reward for all clear types for simplicity

                    # Check if this move would lead to an immediate game over (no subsequent moves possible)
                    # This is a tricky heuristic to calculate perfectly without full lookahead for next pieces.
                    # For now, we'll infer it if, after this move, no pieces from a hypothetical *next* set could be placed.
                    # This is an approximation. A simpler approach is a large penalty if game_state.attempt_placement itself leads to game over.
                    # The GA will primarily learn to avoid game over via the main game score.
                    # We can add a penalty if the *current* placement leads to game_state.check_if_game_over() becoming true
                    # *before* new pieces are generated.
                    
                    
                    # For now, the H_IMMEDIATE_GAME_LOST_PENALTY will be triggered if game_state.game_over is true after the actual placement.
                    # So, it's not part of the *prospective* evaluation here, but rather a post-mortem if the game ends.
                    # A better way would be to simulate one step further if this is critical.
                    # Let's assume the main fitness (game score) handles game over avoidance primarily.
                    # The H_IMMEDIATE_GAME_LOST_PENALTY is more for if a specific move is *known* to be terminal.

                    # temp_gs_for_lookahead = copy.deepcopy(game_state)
                    
                    if sim_game_over:
                        move_outcome_details['game_lost_penalty'] = s_ai.R_GAME_LOST_IMMEDIATE
                    
                    current_move_eval_score = evaluate_potential_move(
                        sim_grid,
                        individual_weights,
                        move_outcome_details
                    )

                    if best_move_details is None or current_move_eval_score > best_move_details[4]: # Index 4 is eval_score
                        best_move_details = (decoded_p_idx, current_piece, decoded_r, decoded_c, current_move_eval_score, action_idx)
            
            if not possible_moves_exist_for_turn or best_move_details is None:
                game_state.game_over = True
                # Apply game lost penalty to the *heuristic evaluation* if we want to consider it here
                # This is complex as it's about the *current* state leading to no moves.
                # The GA learns this by getting low game scores.
                break

            actual_piece_to_place = game_state.current_pieces[best_move_details[0]]
            target_r_actual, target_c_actual = best_move_details[2], best_move_details[3]
            
            placement_results = game_state.attempt_placement(actual_piece_to_place, target_r_actual, target_c_actual)
            
            if not placement_results[0]: # Should not happen
                game_state.game_over = True 
                break
            
            if game_state.game_over : # If placement led to game over (e.g. no more pieces can be placed next)
                # The fitness will be low naturally. The H_IMMEDIATE_GAME_LOST_PENALTY is hard to apply prospectively
                # without knowing the *next* set of pieces. The game score itself is the primary driver.
                pass

        total_score_across_all_games += game_state.score
    return total_score_across_all_games / num_games if num_games > 0 else 0.0

# --- Genetic Operators ---
def selection(population_with_fitness: list[tuple[list[float], float]]) -> list[list[float]]:
    selected_parents = []
    tournament_size = max(2, int(len(population_with_fitness) * 0.1))
    for _ in range(len(population_with_fitness)):
        tournament_contenders = random.sample(population_with_fitness, tournament_size)
        winner = max(tournament_contenders, key=lambda item: item[1])
        selected_parents.append(winner[0])
    return selected_parents

def crossover(parent1: list[float], parent2: list[float]) -> tuple[list[float], list[float]]:
    if random.random() < s_ai.GA_CROSSOVER_RATE:
        alpha = random.random()
        child1_weights = [(alpha * p1_w + (1 - alpha) * p2_w) for p1_w, p2_w in zip(parent1, parent2)]
        beta = random.random()
        child2_weights = [(beta * p2_w + (1 - beta) * p1_w) for p1_w, p2_w in zip(parent1, parent2)]
        return child1_weights, child2_weights
    return list(parent1), list(parent2)

def mutate(individual_weights: list[float]) -> list[float]:
    global current_mutation_rate, current_mutation_strength
    mutated_weights = list(individual_weights)
    for i in range(len(mutated_weights)):
        if random.random() < current_mutation_rate:
            mutation_val = random.uniform(-current_mutation_strength, current_mutation_strength)
            mutated_weights[i] += mutation_val
            mutated_weights[i] = max(-10.0, min(10.0, mutated_weights[i])) # Wider clamp range
    return mutated_weights

# --- Adaptive Mutation Logic ---
def update_adaptive_mutation(generations_since_last_improvement: int):
    global current_mutation_rate, current_mutation_strength
    if generations_since_last_improvement >= s_ai.GA_ADAPTIVE_MUTATION_STAGNATION_THRESHOLD:
        current_mutation_rate = min(s_ai.GA_MUTATION_RATE_MAX, current_mutation_rate + s_ai.GA_ADAPTIVE_MUTATION_RATE_INCREMENT)
        current_mutation_strength = min(s_ai.GA_MUTATION_STRENGTH_MAX, current_mutation_strength + s_ai.GA_ADAPTIVE_MUTATION_STRENGTH_INCREMENT)
        # print(f"Adaptive: Stagnation detected. Increased mutation rate to {current_mutation_rate:.3f}, strength to {current_mutation_strength:.3f}")
    else: # Improvement occurred recently
        current_mutation_rate = max(s_ai.GA_MUTATION_RATE_MIN, current_mutation_rate - s_ai.GA_ADAPTIVE_MUTATION_RATE_DECREMENT)
        current_mutation_strength = max(s_ai.GA_MUTATION_STRENGTH_MIN, current_mutation_strength - s_ai.GA_ADAPTIVE_MUTATION_STRENGTH_DECREMENT)

# --- CSV Logging ---
def setup_logging(log_filename: str):
    header = ['generation', 'best_fitness', 'average_fitness', 'std_dev_fitness', 'mutation_rate', 'mutation_strength']
    for i in range(s_ai.GA_NUM_HEURISTICS):
        header.append(f'best_weight_{i}')
    
    # Create model save directory if it doesn't exist
    if not os.path.exists(s_ai.GA_MODEL_SAVE_DIR):
        os.makedirs(s_ai.GA_MODEL_SAVE_DIR)
        print(f"Created directory: {s_ai.GA_MODEL_SAVE_DIR}")

    # Construct full path for log file within the model save directory
    full_log_path = os.path.join(s_ai.GA_MODEL_SAVE_DIR, log_filename)

    with open(full_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return full_log_path

def log_generation_stats(log_filepath: str, generation: int, best_fitness: float, avg_fitness: float, std_dev_fitness: float, best_individual_weights: list[float]):
    global current_mutation_rate, current_mutation_strength
    row = [generation, f"{best_fitness:.2f}", f"{avg_fitness:.2f}", f"{std_dev_fitness:.2f}", f"{current_mutation_rate:.4f}", f"{current_mutation_strength:.4f}"]
    row.extend([f"{w}" for w in best_individual_weights])
    with open(log_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# --- Main Genetic Algorithm Loop ---
def run_ga():
    global current_mutation_rate, current_mutation_strength # To modify them
    current_mutation_rate = s_ai.GA_MUTATION_RATE_INITIAL
    current_mutation_strength = s_ai.GA_MUTATION_STRENGTH_INITIAL

    # Pygame initialization in the main process (for potential non-GA uses or if GameState expects it)
    pygame_initialized_main = False
    try:
        import pygame
        pygame.init()
        pygame.font.init()
        pygame_initialized_main = True
        print("Pygame initialized successfully in main process for GameState compatibility.")
    except Exception as e:
        print(f"Main process Pygame initialization warning: {e}. Continuing with simulation.")

    timestamp = datetime.now().strftime("%d-%m-%H-%M")
    log_filename = f"training_log_{timestamp}.csv"
    log_filepath = setup_logging(log_filename)
    population = [create_individual() for _ in range(s_ai.GA_POPULATION_SIZE)]
    best_overall_fitness_so_far = -float('inf')
    best_overall_individual_so_far = None
    generations_since_last_improvement = 0
    
    fitness_cache = {} # Initialize fitness cache

    # Determine number of processes for the pool
    # Using cpu_count() can maximize utilization, but might make the system less responsive.
    # Consider using cpu_count() - 1 or a fixed number if needed.
    num_processes = max(1, cpu_count() -1 if cpu_count() > 1 else 1) 
    print(f"Using {num_processes} processes for fitness evaluation.")

    fitness_calculator_with_fixed_games = partial(calculate_fitness, num_games=s_ai.GA_NUM_GAMES_PER_EVALUATION)

    for generation in tqdm(range(1, s_ai.GA_NUM_GENERATIONS + 1), desc="Evolving Generations"):
        fitness_scores_this_gen = [0.0] * len(population) # Initialize with placeholders
        individuals_to_evaluate_indices = []
        individuals_to_evaluate_actual = []

        # Check cache first
        for i, ind in enumerate(population):
            ind_tuple = tuple(ind) # Convert list to tuple for dict key
            if ind_tuple in fitness_cache:
                fitness_scores_this_gen[i] = fitness_cache[ind_tuple]
            else:
                individuals_to_evaluate_indices.append(i)
                individuals_to_evaluate_actual.append(ind)
        
        # Parallelize fitness calculation for non-cached individuals
        if individuals_to_evaluate_actual:
            with Pool(processes=num_processes, initializer=worker_init_pygame) as pool:
                # pool.imap preserves order and works well with tqdm for progress
                calculated_scores = list(tqdm(pool.imap(fitness_calculator_with_fixed_games, individuals_to_evaluate_actual),
                                                    total=len(individuals_to_evaluate_actual),
                                                    desc=f"Gen {generation} Fitness Eval",
                                                    leave=False, unit="indiv"))
            
            # Update fitness_scores_this_gen and cache
            for i, score_idx in enumerate(individuals_to_evaluate_indices):
                fitness_scores_this_gen[score_idx] = calculated_scores[i]
                fitness_cache[tuple(individuals_to_evaluate_actual[i])] = calculated_scores[i]
        
        population_with_fitness = list(zip(population, fitness_scores_this_gen))

        population_with_fitness.sort(key=lambda item: item[1], reverse=True)
        
        current_best_fitness_this_gen = population_with_fitness[0][1]
        current_best_individual_this_gen = population_with_fitness[0][0]
        
        # Calculate average and std dev from the collected fitness_scores_this_gen
        avg_fitness_this_gen = np.mean(fitness_scores_this_gen) if fitness_scores_this_gen else 0
        std_dev_fitness_this_gen = np.std(fitness_scores_this_gen) if fitness_scores_this_gen else 0

        if current_best_fitness_this_gen > best_overall_fitness_so_far:
            best_overall_fitness_so_far = current_best_fitness_this_gen
            best_overall_individual_so_far = list(current_best_individual_this_gen)
            generations_since_last_improvement = 0
            print(f"\nGeneration {generation}: New Overall Best Fitness = {best_overall_fitness_so_far:.2f} (Cache size: {len(fitness_cache)})")
        else:
            generations_since_last_improvement += 1
        print(f"\nGeneration {generation}: Best Fitness = {current_best_fitness_this_gen:.2f}, Avg Fitness: {avg_fitness_this_gen:.2f}, Overall Best: {best_overall_fitness_so_far:.2f} (Cache size: {len(fitness_cache)})")

        log_generation_stats(log_filepath, generation, current_best_fitness_this_gen, avg_fitness_this_gen, std_dev_fitness_this_gen, current_best_individual_this_gen)
        update_adaptive_mutation(generations_since_last_improvement)

        next_generation_population = []
        for i in range(s_ai.GA_ELITISM_COUNT):
            next_generation_population.append(population_with_fitness[i][0])

        selected_parents = selection(population_with_fitness)
        children_needed = s_ai.GA_POPULATION_SIZE - s_ai.GA_ELITISM_COUNT
        children_created = []
        parent_pool_idx = 0
        while len(children_created) < children_needed:
            p1 = selected_parents[parent_pool_idx % len(selected_parents)]
            parent_pool_idx = (parent_pool_idx + 1) % len(selected_parents)
            p2 = selected_parents[parent_pool_idx % len(selected_parents)]
            parent_pool_idx = (parent_pool_idx + 1) % len(selected_parents)
            if p1 is p2 and len(set(map(tuple, selected_parents))) > 1: # Ensure p1 and p2 are different if possible
                # This logic to find a different p2 could be improved for robustness
                # For simplicity, we'll try the next one, but a shuffle or random pick might be better
                # if the parent pool has many identical individuals.
                temp_idx = (parent_pool_idx + random.randint(0, len(selected_parents)-1)) % len(selected_parents)
                p2 = selected_parents[temp_idx]
                # Basic check to ensure p1 and p2 are not the same object if possible
                if p1 is p2 and len(selected_parents) > 1: # If still same and more parents exist
                    p2 = selected_parents[(parent_pool_idx + 1) % len(selected_parents)] # try next one sequentially


            child1, child2 = crossover(p1, p2)
            children_created.append(mutate(child1))
            if len(children_created) < children_needed:
                children_created.append(mutate(child2))
        
        next_generation_population.extend(children_created)
        population = next_generation_population[:s_ai.GA_POPULATION_SIZE]

    print("\n--- Genetic Algorithm Finished ---")
    if best_overall_individual_so_far:
        print(f"Best individual weights found: {[f'{w:.3f}' for w in best_overall_individual_so_far]}")
        print(f"Best fitness score achieved: {best_overall_fitness_so_far:.2f}")
        
        best_weights_filepath = os.path.join(s_ai.GA_MODEL_SAVE_DIR, s_ai.GA_BEST_WEIGHTS_FILE)
        try:
            with open(best_weights_filepath, "w") as f:
                weights_str = ", ".join(map(str, best_overall_individual_so_far))
                f.write(weights_str)
            print(f"Saved best weights to {best_weights_filepath}")
        except IOError:
            print(f"Error: Could not save weights to file: {best_weights_filepath}")
    else:
        print("No best individual found.")

    if pygame_initialized_main and 'pygame' in sys.modules:
        pygame.quit() # Quit Pygame in the main process
    return best_overall_individual_so_far, best_overall_fitness_so_far

if __name__ == "__main__":
    print("Starting Blockdoku Genetic Algorithm Solver...")
    run_ga()
    print("Solver finished.")
