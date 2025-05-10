# blockdoku_env.py
import pygame
import numpy as np
import settings as s # Root AI settings for rewards and AI params
import game.settings as s_game # Game specific settings for display or non-AI game logic
from game.grid import Grid
from game.game_state import GameState
from utils import decode_action, get_valid_action_mask

class BlockdokuEnv:
    def __init__(self, render_mode=None):
        pygame.init()
        pygame.font.init() # Ensure font module is initialized
        self.grid = Grid() 
        self.game = GameState(self.grid)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT))
            pygame.display.set_caption("Blockdoku - AI Player")
            self.clock = pygame.time.Clock()
        self.action_size = s.ACTION_SPACE_SIZE

    def reset(self):
        self.grid = Grid()
        self.game = GameState(self.grid)
        initial_obs = self._get_observation()
        info = self._get_info()
        return initial_obs, info

    # Removed _calculate_isolation method

    def step(self, action_index):
        if self.game.game_over:
            obs = self._get_observation()
            info = self._get_info()
            # Ensure info from last valid state is somewhat sensible for terminal step
            info["lines_cleared"] = 0 
            info["cols_cleared"] = 0
            info["squares_cleared"] = 0
            info["almost_full_count"] = 0
            info["almost_full_details_debug"] = []
            return obs, 0, True, info


        piece_idx, grid_r, grid_c = decode_action(action_index)
        
        reward = 0.0 # RL specific reward
        cleared_r_count, cleared_c_count, cleared_sq_count = 0, 0, 0
        almost_full_count = 0
        almost_full_details_debug = []


        is_move_valid_by_env = False
        piece_to_place = None
        if 0 <= piece_idx < len(self.game.current_pieces):
            piece_to_place = self.game.current_pieces[piece_idx]
            if self.grid.can_place_piece(piece_to_place, grid_r, grid_c):
                is_move_valid_by_env = True
        
        if is_move_valid_by_env:
            # attempt_placement now returns (success, r_cl, c_cl, sq_cl, almost_full_cnt, almost_full_dbg)
            success, r_cl, c_cl, sq_cl, af_cnt, af_dbg = self.game.attempt_placement(piece_to_place, grid_r, grid_c)
            
            cleared_r_count = r_cl
            cleared_c_count = c_cl
            cleared_sq_count = sq_cl
            almost_full_count = af_cnt
            almost_full_details_debug = af_dbg

            if success:
                reward += s.REWARD_BLOCK_PLACED 

                # Reward for "almost full" regions (setup bonus)
                if almost_full_count > 0:
                    # Only reward if it wasn't also a full clear for that region?
                    # The current `get_almost_full_regions_info` checks for 7 or 8, so it won't trigger for 9 (a full clear).
                    reward += almost_full_count * s.REWARD_ALMOST_FULL
                
                num_actual_clear_events = cleared_r_count + cleared_c_count + cleared_sq_count
                if num_actual_clear_events > 0:
                    reward += num_actual_clear_events * s.REWARD_LINE_SQUARE_CLEAR
            else: 
                reward = s.INVALID_MOVE_PENALTY 
        else: 
            reward = s.INVALID_MOVE_PENALTY

        done = self.game.game_over 

        if done:
            reward += s.STUCK_PENALTY_RL 
            
        next_observation = self._get_observation()
        info = self._get_info() # game_score_display is self.game.score

        # Add detailed reward components to info for logging/debugging
        info["lines_cleared"] = cleared_r_count
        info["cols_cleared"] = cleared_c_count
        info["squares_cleared"] = cleared_sq_count
        info["almost_full_count"] = almost_full_count
        # info["almost_full_details_debug"] = almost_full_details_debug # Can be verbose

        return next_observation, reward, done, info

    def _get_info(self):
         return {
             "game_score_display": self.game.score, 
             "available_piece_keys": [p.shape_key for p in self.game.current_pieces],
             "valid_action_mask": get_valid_action_mask(self.game),
             # These will be populated by the step function before returning info
             "lines_cleared": 0, 
             "cols_cleared": 0,
             "squares_cleared": 0,
             "almost_full_count": 0
         }

    def _get_observation(self):
        grid_state = self.grid.get_grid_state() 
        available_pieces_vector = np.zeros(s.NUM_PIECE_TYPES, dtype=np.float32)
        for piece in self.game.current_pieces:
            if piece.shape_key in s.PIECE_KEY_TO_ID:
                piece_id = s.PIECE_KEY_TO_ID[piece.shape_key]
                available_pieces_vector[piece_id] = 1.0
        return {
            "grid": grid_state,
            "pieces": available_pieces_vector
        }

    def render(self, fps=None):
        if self.render_mode != "human" or self.screen is None:
            return
        self.screen.fill(s_game.BLACK) 
        self.grid.draw(self.screen)    
        self.game.draw_score(self.screen) 
        for piece in self.game.current_pieces:
             if not hasattr(piece, 'is_dragging') or not piece.is_dragging:
                piece.draw(self.screen) 
        if self.game.game_over:
            self.game.draw_game_over(self.screen)
        pygame.display.flip()
        if self.clock:
            target_fps = fps if fps is not None else s_game.FPS
            self.clock.tick(target_fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if pygame.get_init(): # Check if pygame is initialized
            pygame.font.quit()
            pygame.quit()
        self.screen = None
        self.clock = None