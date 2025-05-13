# game/game_state.py
import pygame
import random
import game.settings as s_game # Import game-specific settings
from game.piece import Piece
from game.grid import Grid
from typing import List, Optional, Tuple
import copy

class GameState:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.score = 0  # This is the game score
        self.current_pieces: List[Piece] = []
        self.game_over = False
        self._generate_new_pieces()
        # Ensure font is initialized after pygame.init() if it's called globally
        if pygame.get_init() and pygame.font.get_init(): # Check both pygame and font module
            self.font = pygame.font.SysFont(None, 36)
        else:
            self.font = None 


    def _generate_new_pieces(self):
        """Generates NUM_PIECES_AVAILABLE new random pieces."""
        self.current_pieces.clear()
        # Ensure PIECE_DEFINITIONS is accessed correctly, might need s_game
        available_keys = list(s_game.PIECE_DEFINITIONS.keys())
        for i in range(s_game.NUM_PIECES_AVAILABLE):
            shape_key = random.choice(available_keys)
            piece = Piece(shape_key, s_game.PIECE_DEFINITIONS[shape_key])
            piece.set_origin(s_game.PIECE_AREA_X, s_game.PIECE_AREA_Y + i * s_game.PIECE_SPACING)
            self.current_pieces.append(piece)
        self.check_if_game_over()


    def attempt_placement(self, piece: Piece, grid_r: int, grid_c: int) -> Tuple[bool, int, int, int, int, List[str]]:
        """
        Tries to place a piece, updates score, gets clear counts and almost full counts.
        Returns: Tuple (success_bool, cleared_r, cleared_c, cleared_sq, almost_full_new_count, almost_full_details_debug)
        """
        cleared_r_count, cleared_c_count, cleared_sq_count = 0, 0, 0
        almost_full_new_count = 0
        almost_full_details_debug = []

        if self.game_over:
            return False, cleared_r_count, cleared_c_count, cleared_sq_count, almost_full_new_count, almost_full_details_debug

        if self.grid.can_place_piece(piece, grid_r, grid_c):
            # Get cells of the piece to be placed for "almost full" calculation
            piece_cells_on_grid = set()
            for r_offset, c_offset in piece.relative_cells:
                piece_cells_on_grid.add((grid_r + r_offset, grid_c + c_offset))

            self.grid.place_piece(piece, grid_r, grid_c) # Piece is now on the grid

            # --- Game Score Update ---
            self.score += piece.get_block_count() * s_game.SCORE_PER_BLOCK_GAME

            # --- Calculate "Almost Full" bonus info (based on state *after* placement, *before* clears) ---
            # This simple version counts how many relevant regions are now almost full.
            # A more advanced version would compare before/after this piece for *newly* almost full.
            almost_full_new_count, almost_full_details_debug = self.grid.get_almost_full_regions_info(piece_cells_on_grid)

            # --- Clear lines and Score for Clears ---
            cleared_r_count, cleared_c_count, cleared_sq_count = self.grid.clear_lines_and_squares()
            self.score += cleared_r_count * s_game.SCORE_PER_LINE_GAME
            self.score += cleared_c_count * s_game.SCORE_PER_LINE_GAME 
            self.score += cleared_sq_count * s_game.SCORE_PER_SQUARE_GAME

            self.current_pieces.remove(piece)

            if not self.current_pieces:
                self._generate_new_pieces()
            else:
                 self.check_if_game_over()

            return True, cleared_r_count, cleared_c_count, cleared_sq_count, almost_full_new_count, almost_full_details_debug
        else:
            return False, 0, 0, 0, 0, []


    def check_if_game_over(self):
        """Checks if any of the current pieces can be placed anywhere."""
        if not self.current_pieces: # Should not happen if logic is correct (new pieces generated)
             self.game_over = False 
             return

        can_place_any = False
        for piece in self.current_pieces:
            for r in range(self.grid.height): # Use grid dimensions from grid object
                for c in range(self.grid.width):
                    if self.grid.can_place_piece(piece, r, c):
                        can_place_any = True
                        break 
                if can_place_any:
                    break 
            if can_place_any:
                break 

        if not can_place_any:
            self.game_over = True
    
    def draw_score(self, surface):
        if self.font: 
            score_text = self.font.render(f"Game Score: {self.score}", True, s_game.WHITE)
            score_rect = score_text.get_rect(topleft=(s_game.PIECE_AREA_X, 10))
            surface.blit(score_text, score_rect)

    def draw_available_pieces(self, surface):
        for piece in self.current_pieces:
            if not hasattr(piece, 'is_dragging') or not piece.is_dragging: 
                piece.draw(surface)

    def draw_game_over(self, surface):
         if self.game_over and self.font: 
            overlay = pygame.Surface((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)) 
            surface.blit(overlay, (0, 0))

            go_font = pygame.font.SysFont(None, 72)
            go_text = go_font.render("Game Over!", True, s_game.WHITE)
            go_rect = go_text.get_rect(center=(s_game.SCREEN_WIDTH / 2, s_game.SCREEN_HEIGHT / 2 - 30))
            surface.blit(go_text, go_rect)

            score_font = pygame.font.SysFont(None, 48)
            final_score_text = score_font.render(f"Final Game Score: {self.score}", True, s_game.WHITE)
            final_score_rect = final_score_text.get_rect(center=(s_game.SCREEN_WIDTH / 2, s_game.SCREEN_HEIGHT / 2 + 30))
            surface.blit(final_score_text, final_score_rect)
            
    def get_state_for_ai(self) -> dict:
        """Provides the current game state information needed for an AI."""
        return {
            "grid_state": self.grid.get_grid_state(), 
            "current_pieces": self.current_pieces, 
            "available_piece_keys": [p.shape_key for p in self.current_pieces],
            "score": self.score, # This is the simple game score
            "game_over": self.game_over
        }
    
    def simulate_attempt_placement(self, piece: Piece, grid_r: int, grid_c: int) -> Tuple[bool, int, int, int, int, List[str], bool]:
        """
        Simulates placing a piece to get clear counts, almost full counts, and if the game would end next.
        Does NOT modify the actual game state (grid, score, pieces).
        Returns: Tuple (
            success_bool,
            cleared_r,
            cleared_c,
            cleared_sq,
            almost_full_new_count,
            almost_full_details_debug,
            would_game_over_next_turn_bool
        )
        """
        cleared_r_count, cleared_c_count, cleared_sq_count = 0, 0, 0
        almost_full_new_count = 0
        almost_full_details_debug = []
        would_game_over_next_turn = False # Default to False

        if self.grid.can_place_piece(piece, grid_r, grid_c):
            # print("SIMUATION")
            sim_grid = copy.deepcopy(self.grid)

            piece_cells_on_grid = set()
            for r_offset, c_offset in piece.relative_cells:
                piece_cells_on_grid.add((grid_r + r_offset, grid_c + c_offset))

            sim_grid.place_piece(piece, grid_r, grid_c)
            almost_full_new_count, almost_full_details_debug = sim_grid.get_almost_full_regions_info(piece_cells_on_grid)
            cleared_r_count, cleared_c_count, cleared_sq_count = sim_grid.clear_lines_and_squares() # Modifies sim_grid

            # Determine if this is the last piece from the current set
            is_last_current_piece = len(self.current_pieces) == 1 and self.current_pieces[0] is piece
            # Check if any of the current pieces can be placed anywhere
            

            can_place_any = False        
            if not is_last_current_piece:   
                
                for piece in self.current_pieces:
                    for r in range(self.grid.height): # Use grid dimensions from grid object
                        for c in range(self.grid.width):
                            if sim_grid.can_place_piece(piece, r, c):
                                can_place_any = True
                                break 
                        if can_place_any:
                            break 
                    if can_place_any:
                        break 

            if not can_place_any:
                
                would_game_over_next_turn = True

            return cleared_r_count, cleared_c_count, cleared_sq_count, almost_full_new_count, would_game_over_next_turn,sim_grid
        else:
            
            return 0, 0, 0, 0, False, False # Move not possible, so next turn game over is False by this path
