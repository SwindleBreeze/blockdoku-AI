# genetic_solver.py
import random
import copy
import sys
import os
import statistics
import pickle
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
# from tqdm import tqdm # Standard tqdm
from tqdm.auto import tqdm # More robust tqdm for different environments

# --- Game Imports ---
try:
    from game.game_state import GameState
    from game.grid import Grid
    # Piece and game.settings are used implicitly by GameState/Grid
except ImportError as e:
    print(f"Error importing game modules: {e}")
    print("Please ensure 'genetic_solver.py' is in the parent directory of the 'game' package,")
    print("and that 'utils.py' and your AI 'settings.py' are in the same directory as 'genetic_solver.py'.")
    sys.exit(1)

# --- AI Utils and Settings Imports ---
try:
    import utils # Your utils.py
    import settings as s_ai # Your AI settings file (e.g., blockdoku/settings.py or a root settings.py for AI)
except ImportError as e:
    print(f"Error importing 'utils.py' or AI 'settings.py': {e}")
    print("Ensure 'utils.py' and your AI settings file (e.g., 'settings.py' aliased as s_ai)")
    print("are in the same directory as 'genetic_solver.py'.")
    sys.exit(1)

# --- Default GA Parameters (will be configurable) ---
DEFAULT_POPULATION_SIZE = 30
DEFAULT_NUM_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.15
DEFAULT_MUTATION_STRENGTH = 0.3
DEFAULT_CROSSOVER_RATE = 0.7
DEFAULT_ELITISM_COUNT = 2
DEFAULT_NUM_GAMES_PER_EVALUATION = 1
DEFAULT_SELECTION_METHOD = "tournament"
DEFAULT_TOURNAMENT_SIZE_PERCENT = 0.1
DEFAULT_CHECKPOINT_FREQUENCY = 5
DEFAULT_NICHING_ENABLED = False
DEFAULT_SIGMA_SHARE = 0.2
DEFAULT_PARALLEL_ENABLED = True
DEFAULT_ADAPTIVE_PARAMS_ENABLED = True

# Heuristic indices for GA's chromosome (weights)
H_CLEARED_LINES_AND_COLUMNS = 0
H_CLEARED_3X3_SQUARES = 1
H_AGGREGATE_HEIGHT = 2
H_NUM_HOLES = 3
H_BUMPINESS = 4
H_ALMOST_FULL_REGIONS = 5
NUM_HEURISTICS = 6

# Global fitness cache
fitness_cache = {}

# --- Command Line Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Blockdoku Genetic Algorithm")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POPULATION_SIZE, help="Population size")
    parser.add_argument("--generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE, help="Mutation rate")
    parser.add_argument("--mutation-strength", type=float, default=DEFAULT_MUTATION_STRENGTH, help="Mutation strength")
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE, help="Crossover rate")
    parser.add_argument("--elitism-count", type=int, default=DEFAULT_ELITISM_COUNT, help="Number of elite individuals")
    parser.add_argument("--games-per-eval", type=int, default=DEFAULT_NUM_GAMES_PER_EVALUATION, help="Games per fitness evaluation")
    parser.add_argument("--selection", type=str, choices=["tournament", "rank"], default=DEFAULT_SELECTION_METHOD, help="Selection method")
    parser.add_argument("--tournament-size-percent", type=float, default=DEFAULT_TOURNAMENT_SIZE_PERCENT, help="Tournament size as percentage of population")
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQUENCY, help="Checkpoint frequency (generations)")
    parser.add_argument("--niching", action="store_true", default=DEFAULT_NICHING_ENABLED, help="Enable niching")
    parser.add_argument("--sigma-share", type=float, default=DEFAULT_SIGMA_SHARE, help="Sigma sharing parameter for niching")
    parser.add_argument("--parallel", action="store_true", default=DEFAULT_PARALLEL_ENABLED, help="Enable parallel fitness evaluation")
    parser.add_argument("--adaptive-params", action="store_true", default=DEFAULT_ADAPTIVE_PARAMS_ENABLED, help="Enable adaptive parameters")
    parser.add_argument("--load-checkpoint", type=str, help="Load from checkpoint file")
    return parser.parse_args()

# --- Helper Functions for Calculating Heuristics (same as before) ---
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

# --- Genetic Algorithm Class ---
class BlockdokuGeneticAlgorithm:
    def __init__(self, args):
        self.population_size = args.pop_size
        self.num_generations = args.generations
        self.base_mutation_rate = args.mutation_rate
        self.current_mutation_rate = args.mutation_rate
        self.mutation_strength = args.mutation_strength
        self.base_crossover_rate = args.crossover_rate
        self.current_crossover_rate = args.crossover_rate
        self.elitism_count = args.elitism_count
        self.num_games_per_evaluation = args.games_per_eval
        self.selection_method = args.selection
        self.tournament_size_percent = args.tournament_size_percent
        self.checkpoint_frequency = args.checkpoint_freq
        self.niching_enabled = args.niching
        self.sigma_share = args.sigma_share
        self.parallel_enabled = args.parallel
        self.adaptive_params_enabled = args.adaptive_params
        
        # Initialize population and state
        self.population = []
        self.generation = 0
        self.best_overall_fitness = -float('inf')
        self.best_overall_individual = None
        
        # Initialize or load from checkpoint
        if args.load_checkpoint:
            self.load_checkpoint(args.load_checkpoint)
        else:
            self.population = [self.create_individual() for _ in range(self.population_size)]
        
        # Initialize pygame for GameState compatibility
        self.pygame_initialized = self.init_pygame()
        
    def init_pygame(self):
        try:
            import pygame
            pygame.init()
            pygame.font.init()
            print("Pygame initialized successfully for GameState compatibility.")
            return True
        except Exception as e:
            print(f"Pygame initialization warning: {e}. Continuing with simulation.")
            return False
    
    def cleanup_pygame(self):
        if self.pygame_initialized and 'pygame' in sys.modules:
            import pygame
            pygame.quit()
    
    def create_individual(self) -> list[float]:
        return [random.uniform(-1.0, 1.0) for _ in range(NUM_HEURISTICS)]
    
    def evaluate_potential_move(
        self,
        grid_state_after_placement_before_clears: Grid,
        weights: list[float],
        outcome_stats: dict
    ) -> float:
        score = 0.0
        agg_height = calculate_aggregate_height(grid_state_after_placement_before_clears)
        num_holes = calculate_num_holes(grid_state_after_placement_before_clears)
        bumpiness_val = calculate_bumpiness(grid_state_after_placement_before_clears)

        score += weights[H_AGGREGATE_HEIGHT] * agg_height
        score += weights[H_NUM_HOLES] * num_holes
        score += weights[H_BUMPINESS] * bumpiness_val
        score += weights[H_CLEARED_LINES_AND_COLUMNS] * outcome_stats['cleared_lines_cols']
        score += weights[H_CLEARED_3X3_SQUARES] * outcome_stats['cleared_squares']
        score += weights[H_ALMOST_FULL_REGIONS] * outcome_stats['almost_full_regions']
        return score
    
    def calculate_fitness(self, individual_weights: list[float], num_games: int) -> float:
        # Check if this individual has been evaluated before
        individual_tuple = tuple(individual_weights)
        if individual_tuple in fitness_cache:
            return fitness_cache[individual_tuple]
        
        total_score_across_all_games = 0.0

        for _ in range(num_games):
            game_grid = Grid()
            game_state = GameState(game_grid)

            while not game_state.game_over:
                best_move_details = None
                
                if not game_state.current_pieces:
                    game_state._generate_new_pieces()
                    if game_state.game_over: break
                
                valid_actions_mask = utils.get_valid_action_mask(game_state)
                possible_moves_exist_for_turn = False

                for action_idx in range(s_ai.ACTION_SPACE_SIZE):
                    if valid_actions_mask[action_idx]:
                        possible_moves_exist_for_turn = True
                        
                        decoded_p_idx, decoded_r, decoded_c = utils.decode_action(action_idx)

                        if decoded_p_idx >= len(game_state.current_pieces):
                            continue
                            
                        current_piece = game_state.current_pieces[decoded_p_idx]
                        
                        sim_grid_post_placement_pre_clear = copy.deepcopy(game_grid)
                        sim_grid_post_placement_pre_clear.place_piece(current_piece, decoded_r, decoded_c)

                        piece_cells_on_grid = set()
                        for r_offset, c_offset in current_piece.relative_cells:
                            piece_cells_on_grid.add((decoded_r + r_offset, decoded_c + c_offset))
                        
                        af_count, _ = sim_grid_post_placement_pre_clear.get_almost_full_regions_info(piece_cells_on_grid)

                        temp_grid_for_clear_counting = copy.deepcopy(sim_grid_post_placement_pre_clear)
                        cleared_r_count, cleared_c_count, cleared_sq_count = temp_grid_for_clear_counting.clear_lines_and_squares()
                        
                        move_outcome_stats = {
                            'cleared_lines_cols': cleared_r_count + cleared_c_count,
                            'cleared_squares': cleared_sq_count,
                            'almost_full_regions': af_count,
                            'blocks_in_piece': current_piece.get_block_count()
                        }
                        
                        current_move_eval_score = self.evaluate_potential_move(
                            sim_grid_post_placement_pre_clear,
                            individual_weights,
                            move_outcome_stats
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
            
            total_score_across_all_games += game_state.score

        fitness = total_score_across_all_games / num_games if num_games > 0 else 0.0
        
        # Cache the result
        fitness_cache[individual_tuple] = fitness
        return fitness
    
    def evaluate_population_parallel(self, population):
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                lambda individual: self.calculate_fitness(individual, self.num_games_per_evaluation),
                population
            ))
        return results
    
    def evaluate_population(self):
        population_with_fitness = []
        
        if self.parallel_enabled:
            fitness_values = self.evaluate_population_parallel(self.population)
            population_with_fitness = list(zip(self.population, fitness_values))
        else:
            for individual in tqdm(self.population, desc=f"Gen {self.generation + 1} Fitness Eval", leave=False, unit="indiv"):
                fitness = self.calculate_fitness(individual, self.num_games_per_evaluation)
                population_with_fitness.append((individual, fitness))
                
        return population_with_fitness
    
    def tournament_selection(self, population_with_fitness):
        selected_parents = []
        tournament_size = max(2, int(len(population_with_fitness) * self.tournament_size_percent))
        for _ in range(len(population_with_fitness)):
            tournament_contenders = random.sample(population_with_fitness, tournament_size)
            winner = max(tournament_contenders, key=lambda item: item[1])
            selected_parents.append(winner[0])
        return selected_parents
    
    def rank_selection(self, population_with_fitness):
        # Sort by fitness
        sorted_pop = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
        # Assign selection probabilities based on rank
        total = len(sorted_pop) * (len(sorted_pop) + 1) / 2
        selected = []
        
        for _ in range(len(sorted_pop)):
            r = random.random() * total
            cum_prob = 0
            for i, (individual, _) in enumerate(sorted_pop):
                rank_weight = len(sorted_pop) - i
                cum_prob += rank_weight
                if cum_prob >= r:
                    selected.append(individual)
                    break
        
        return selected
    
    def selection(self, population_with_fitness):
        if self.selection_method == "tournament":
            return self.tournament_selection(population_with_fitness)
        elif self.selection_method == "rank":
            return self.rank_selection(population_with_fitness)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def crossover(self, parent1, parent2):
        if random.random() < self.current_crossover_rate:
            alpha = random.random()
            child1_weights = [(alpha * p1_w + (1 - alpha) * p2_w) for p1_w, p2_w in zip(parent1, parent2)]
            beta = random.random()
            child2_weights = [(beta * p2_w + (1 - beta) * p1_w) for p1_w, p2_w in zip(parent1, parent2)]
            return child1_weights, child2_weights
        return list(parent1), list(parent2)
    
    def mutate(self, individual_weights):
        mutated_weights = list(individual_weights)
        for i in range(len(mutated_weights)):
            if random.random() < self.current_mutation_rate:
                mutation_val = random.uniform(-self.mutation_strength, self.mutation_strength)
                mutated_weights[i] += mutation_val
                mutated_weights[i] = max(-5.0, min(5.0, mutated_weights[i]))
        return mutated_weights
    
    def calculate_population_diversity(self):
        # Calculate standard deviation across each gene position
        if not self.population:
            return 0.0
        
        gene_stds = []
        for i in range(NUM_HEURISTICS):
            gene_values = [ind[i] for ind in self.population]
            if len(gene_values) > 1:  # Need at least 2 values for std
                gene_stds.append(statistics.stdev(gene_values))
            else:
                gene_stds.append(0.0)
                
        return sum(gene_stds) / len(gene_stds) if gene_stds else 0.0
    
    def update_adaptive_parameters(self):
        if not self.adaptive_params_enabled:
            return
            
        diversity = self.calculate_population_diversity()
        # Scale mutation rate: increase when diversity is low
        diversity_factor = max(0.5, min(2.0, 1.0 + (0.5 - diversity) * 2))
        self.current_mutation_rate = min(0.5, self.base_mutation_rate * diversity_factor)
        
        # For crossover, we might want the opposite effect (more crossover when diversity is high)
        self.current_crossover_rate = min(1.0, self.base_crossover_rate * (1.0 + diversity * 0.5))
    
    def calculate_crowding_distance(self, individual1, individual2):
        return sum(abs(g1 - g2) for g1, g2 in zip(individual1, individual2))
    
    def apply_niching(self, population_with_fitness):
        if not self.niching_enabled:
            return population_with_fitness
            
        niched_fitness = []
        for i, (ind1, fit1) in enumerate(population_with_fitness):
            niche_count = 0
            for j, (ind2, _) in enumerate(population_with_fitness):
                if i != j:
                    distance = self.calculate_crowding_distance(ind1, ind2)
                    if distance < self.sigma_share:
                        niche_count += 1 - (distance / self.sigma_share)
            shared_fitness = fit1 / (1 + niche_count) if niche_count > 0 else fit1
            niched_fitness.append((ind1, shared_fitness))
        return niched_fitness
    
    def save_checkpoint(self):
        checkpoint = {
            "generation": self.generation,
            "population": self.population,
            "best_individual": self.best_overall_individual,
            "best_fitness": self.best_overall_fitness,
            "parameters": {
                "population_size": self.population_size,
                "num_generations": self.num_generations,
                "mutation_rate": self.base_mutation_rate,
                "mutation_strength": self.mutation_strength,
                "crossover_rate": self.base_crossover_rate,
                "elitism_count": self.elitism_count,
                "num_games_per_evaluation": self.num_games_per_evaluation,
                "selection_method": self.selection_method,
                "tournament_size_percent": self.tournament_size_percent,
                "niching_enabled": self.niching_enabled,
                "sigma_share": self.sigma_share,
                "parallel_enabled": self.parallel_enabled,
                "adaptive_params_enabled": self.adaptive_params_enabled
            }
        }
        
        filename = f"ga_checkpoint_gen_{self.generation}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)
            
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        try:
            with open(filename, "rb") as f:
                checkpoint = pickle.load(f)
                
            self.generation = checkpoint["generation"]
            self.population = checkpoint["population"]
            self.best_overall_individual = checkpoint["best_individual"]
            self.best_overall_fitness = checkpoint["best_fitness"]
            
            # Optionally load parameters if they exist in the checkpoint
            if "parameters" in checkpoint:
                params = checkpoint["parameters"]
                self.population_size = params.get("population_size", self.population_size)
                self.num_generations = params.get("num_generations", self.num_generations)
                self.base_mutation_rate = params.get("mutation_rate", self.base_mutation_rate)
                self.current_mutation_rate = self.base_mutation_rate
                self.mutation_strength = params.get("mutation_strength", self.mutation_strength)
                self.base_crossover_rate = params.get("crossover_rate", self.base_crossover_rate)
                self.current_crossover_rate = self.base_crossover_rate
                self.elitism_count = params.get("elitism_count", self.elitism_count)
                self.num_games_per_evaluation = params.get("num_games_per_evaluation", self.num_games_per_evaluation)
                self.selection_method = params.get("selection_method", self.selection_method)
                self.tournament_size_percent = params.get("tournament_size_percent", self.tournament_size_percent)
                self.niching_enabled = params.get("niching_enabled", self.niching_enabled)
                self.sigma_share = params.get("sigma_share", self.sigma_share)
                self.parallel_enabled = params.get("parallel_enabled", self.parallel_enabled)
                self.adaptive_params_enabled = params.get("adaptive_params_enabled", self.adaptive_params_enabled)
                
            print(f"Loaded checkpoint from generation {self.generation}")
            print(f"Best fitness so far: {self.best_overall_fitness}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting with a fresh population")
            self.population = [self.create_individual() for _ in range(self.population_size)]
    
    def run(self):
        try:
            for gen in tqdm(range(self.generation, self.num_generations), desc="Evolving Generations"):
                self.generation = gen
                
                # Update adaptive parameters based on population diversity
                self.update_adaptive_parameters()
                
                # Evaluate population
                population_with_fitness = self.evaluate_population()
                
                # Apply niching if enabled
                if self.niching_enabled:
                    population_with_fitness = self.apply_niching(population_with_fitness)
                
                # Sort by fitness
                population_with_fitness.sort(key=lambda item: item[1], reverse=True)
                
                # Get best individual for this generation
                current_best_fitness = population_with_fitness[0][1]
                current_best_individual = population_with_fitness[0][0]
                
                # Update best overall
                if current_best_fitness > self.best_overall_fitness:
                    self.best_overall_fitness = current_best_fitness
                    self.best_overall_individual = list(current_best_individual)
                
                # Print progress
                print(f"\nGeneration {gen + 1}: Best Fitness = {current_best_fitness:.2f}, Overall Best: {self.best_overall_fitness:.2f}")
                if self.adaptive_params_enabled:
                    diversity = self.calculate_population_diversity()
                    print(f"Population diversity: {diversity:.4f}, Mutation rate: {self.current_mutation_rate:.4f}, Crossover rate: {self.current_crossover_rate:.4f}")
                
                # Save checkpoint if needed
                if self.checkpoint_frequency > 0 and (gen + 1) % self.checkpoint_frequency == 0:
                    self.save_checkpoint()
                
                # Create next generation
                next_generation_population = []
                
                # Elitism - copy best individuals unchanged
                for i in range(min(self.elitism_count, len(population_with_fitness))):
                    next_generation_population.append(population_with_fitness[i][0])
                
                # Selection, crossover, and mutation for the rest
                selected_parents = self.selection(population_with_fitness)
                children_needed = self.population_size - len(next_generation_population)
                children_created = []
                parent_pool_idx = 0
                
                while len(children_created) < children_needed:
                    p1 = selected_parents[parent_pool_idx % len(selected_parents)]
                    parent_pool_idx = (parent_pool_idx + 1) % len(selected_parents)
                    p2 = selected_parents[parent_pool_idx % len(selected_parents)]
                    parent_pool_idx = (parent_pool_idx + 1) % len(selected_parents)
                    
                    # Ensure p1 and p2 are different if possible
                    if p1 is p2 and len(set(map(tuple, selected_parents))) > 1:
                        temp_idx = (parent_pool_idx + 1) % len(selected_parents)
                        p2 = selected_parents[temp_idx]
                    
                    child1, child2 = self.crossover(p1, p2)
                    children_created.append(self.mutate(child1))
                    if len(children_created) < children_needed:
                        children_created.append(self.mutate(child2))
                
                next_generation_population.extend(children_created)
                self.population = next_generation_population[:self.population_size]
            
            # Save final checkpoint
            self.save_checkpoint()
            
            # Final output
            print("\n--- Genetic Algorithm Finished ---")
            if self.best_overall_individual:
                print(f"Best individual weights found: {[f'{w:.3f}' for w in self.best_overall_individual]}")
                print(f"Best fitness score achieved: {self.best_overall_fitness:.2f}")
                try:
                    with open("best_blockdoku_ga_weights.txt", "w") as f:
                        weights_str = ", ".join(map(str, self.best_overall_individual))
                        f.write(weights_str)
                    print("Saved best weights to best_blockdoku_ga_weights.txt")
                except IOError:
                    print("Error: Could not save weights to file.")
            else:
                print("No best individual found.")
                
            return self.best_overall_individual, self.best_overall_fitness
            
        finally:
            # Cleanup
            self.cleanup_pygame()

# --- Main Entry Point ---
def main():
    print("Starting Blockdoku Genetic Algorithm Solver...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Initialize and run the genetic algorithm
    ga = BlockdokuGeneticAlgorithm(args)
    best_individual, best_fitness = ga.run()
    
    print("Solver finished.")
    return best_individual, best_fitness

if __name__ == "__main__":
    main()