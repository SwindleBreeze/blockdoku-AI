# blockdoku_env.py
import pygame
import numpy as np
import settings as s
from grid import Grid
from game_state import GameState
from utils import decode_action, get_valid_action_mask # Import masking util

class BlockdokuEnv:
    """ A wrapper for the Blockdoku game to make it compatible with RL agents. """

    def __init__(self, render_mode=None):
        pygame.init()
        self.grid = Grid()
        self.game = GameState(self.grid) # GameState manages score, pieces, etc.
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((s.SCREEN_WIDTH, s.SCREEN_HEIGHT))
            pygame.display.set_caption("Blockdoku - AI Player")
            self.clock = pygame.time.Clock()
        # Define action space size
        self.action_size = s.ACTION_SPACE_SIZE

    def reset(self):
        """ Resets the environment to the starting state. """
        self.grid = Grid()
        self.game = GameState(self.grid)
        initial_obs = self._get_observation() # Returns dict
        info = self._get_info()
        return initial_obs, info

    def step(self, action_index):
        """ Executes one action in the environment. """
        if self.game.game_over:
            # Should ideally not happen if agent checks 'done' flag
            return self._get_observation(), 0, True, self._get_info()

        prev_score = self.game.score
        piece_idx, grid_r, grid_c = decode_action(action_index)

        # --- Action Validity Check ---
        # Check if the chosen piece index is valid and the move is possible
        is_move_valid = False
        current_pieces = self.game.current_pieces
        if 0 <= piece_idx < len(current_pieces):
            piece_to_place = current_pieces[piece_idx]
            if self.grid.can_place_piece(piece_to_place, grid_r, grid_c):
                 is_move_valid = True

        # --- Execute Move if Valid ---
        success = False
        if is_move_valid:
             # Use the existing GameState method which handles placing, clearing, scoring
             # Note: attempt_placement modifies current_pieces if successful
             success = self.game.attempt_placement(piece_to_place, grid_r, grid_c)

        # --- Calculate Reward ---
        # Basic reward: score difference. Penalize invalid moves?
        reward = 0
        if success:
             # Scale the score difference
             reward = (self.game.score - prev_score) / s.REWARD_SCALING_FACTOR
        elif not is_move_valid:
             # Use the scaled invalid move penalty from settings
            reward = s.INVALID_MOVE_PENALTY

        # --- Get Next State and Done Flag ---
        done = self.game.game_over
        next_observation = self._get_observation() # Return the new state dict
        info = self._get_info() # Get info like current score, valid moves mask

        return next_observation, reward, done, info

    def _get_observation(self):
        """ Returns the composite observation: grid + available pieces vector."""
        grid_state = self.grid.get_grid_state() # (H, W, C) numpy array

        # Create multi-hot vector for available pieces
        # Size NUM_PIECE_TYPES, value 1 if piece type is available, 0 otherwise
        available_pieces_vector = np.zeros(s.NUM_PIECE_TYPES, dtype=np.float32)
        for piece in self.game.current_pieces:
            if piece.shape_key in s.PIECE_KEY_TO_ID:
                piece_id = s.PIECE_KEY_TO_ID[piece.shape_key]
                available_pieces_vector[piece_id] = 1.0
            else:
                print(f"Warning: Piece key '{piece.shape_key}' not found in PIECE_KEY_TO_ID mapping.")

        # Return as a dictionary
        return {
            "grid": grid_state,
            "pieces": available_pieces_vector
        }

    def _get_info(self):
         """ Returns supplementary info (not used for training directly). """
         # Include the valid action mask here!
         return {
             "score": self.game.score,
             "available_piece_keys": [p.shape_key for p in self.game.current_pieces],
             "valid_action_mask": get_valid_action_mask(self.game) # Crucial for better agents
         }

    def render(self, fps=None):
        """ Renders the current game state using Pygame. """
        if self.render_mode != "human" or self.screen is None:
            return

        # --- Drawing Logic (adapted from main.py) ---
        self.screen.fill(s.BLACK)
        self.grid.draw(self.screen)
        self.game.draw_score(self.screen)
        # Draw available pieces (ensure they are drawn if not being 'dragged')
        for piece in self.game.current_pieces:
            piece.draw(self.screen) # Assuming piece stores its selection area pos

        if self.game.game_over:
            self.game.draw_game_over(self.screen)

        pygame.display.flip()
        if self.clock:
            target_fps = fps if fps is not None else s.PLAY_FPS
            self.clock.tick(target_fps)

        # Handle Pygame events like closing the window during rendering
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                # Need a way to signal training/play loop to stop
                # raise SystemExit("Pygame window closed")


    def close(self):
        """ Cleans up Pygame resources. """
        pygame.quit()
        self.screen = None
        self.clock = None