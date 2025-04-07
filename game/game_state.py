# game_state.py
import pygame
import random
import settings as s
from game.piece import Piece
from game.grid import Grid
from typing import List, Optional, Tuple

class GameState:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.score = 0
        self.current_pieces: List[Piece] = []
        self.game_over = False
        self._generate_new_pieces()
        self.font = pygame.font.SysFont(None, 36) # Font for score

    def _generate_new_pieces(self):
        """Generates NUM_PIECES_AVAILABLE new random pieces."""
        self.current_pieces.clear()
        available_keys = list(s.PIECE_DEFINITIONS.keys())
        for i in range(s.NUM_PIECES_AVAILABLE):
            shape_key = random.choice(available_keys)
            piece = Piece(shape_key, s.PIECE_DEFINITIONS[shape_key])
            # Position pieces in the selection area
            piece.set_origin(s.PIECE_AREA_X, s.PIECE_AREA_Y + i * s.PIECE_SPACING)
            self.current_pieces.append(piece)
        # After generating, immediately check if game is over
        self.check_if_game_over()


    def attempt_placement(self, piece: Piece, grid_r: int, grid_c: int) -> bool:
        """Tries to place a piece, updates score, clears lines, checks game over."""
        if self.game_over:
            return False

        if self.grid.can_place_piece(piece, grid_r, grid_c):
            self.grid.place_piece(piece, grid_r, grid_c)

            # Score for placing the piece
            self.score += piece.get_block_count() * s.SCORE_PER_BLOCK

            # Clear lines and score
            cleared_r, cleared_c, cleared_sq = self.grid.clear_lines_and_squares()
            self.score += (cleared_r + cleared_c + cleared_sq) * s.SCORE_PER_LINE # Use same bonus for row/col/square

            # Remove piece from available list
            self.current_pieces.remove(piece)

            # Generate new pieces if needed
            if not self.current_pieces:
                self._generate_new_pieces()
            else:
                 # Even if pieces remain, check if game is over *now*
                 self.check_if_game_over()

            return True
        else:
            return False

    def check_if_game_over(self):
        """Checks if any of the current pieces can be placed anywhere."""
        if not self.current_pieces: # Should not happen if logic is correct, but safe check
             self.game_over = False # Can always generate new pieces
             return

        can_place_any = False
        for piece in self.current_pieces:
            for r in range(self.grid.height):
                for c in range(self.grid.width):
                    if self.grid.can_place_piece(piece, r, c):
                        can_place_any = True
                        break # Found a valid spot for this piece
                if can_place_any:
                    break # Found a valid spot for this piece
            if can_place_any:
                break # Found a piece that can be placed

        if not can_place_any:
            #print("Game Over Condition Met!")
            self.game_over = True


    def draw_score(self, surface):
        score_text = self.font.render(f"Score: {self.score}", True, s.WHITE)
        score_rect = score_text.get_rect(topleft=(s.PIECE_AREA_X, 10))
        surface.blit(score_text, score_rect)

    def draw_available_pieces(self, surface):
        for piece in self.current_pieces:
             # Only draw if not currently being dragged (handled separately in main loop)
            if not piece.is_dragging:
                piece.draw(surface)

    def draw_game_over(self, surface):
         if self.game_over:
            overlay = pygame.Surface((s.SCREEN_WIDTH, s.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)) # Semi-transparent black overlay
            surface.blit(overlay, (0, 0))

            go_font = pygame.font.SysFont(None, 72)
            go_text = go_font.render("Game Over!", True, s.WHITE)
            go_rect = go_text.get_rect(center=(s.SCREEN_WIDTH / 2, s.SCREEN_HEIGHT / 2 - 30))
            surface.blit(go_text, go_rect)

            score_font = pygame.font.SysFont(None, 48)
            final_score_text = score_font.render(f"Final Score: {self.score}", True, s.WHITE)
            final_score_rect = final_score_text.get_rect(center=(s.SCREEN_WIDTH / 2, s.SCREEN_HEIGHT / 2 + 30))
            surface.blit(final_score_text, final_score_rect)

    # --- AI Interaction Methods ---

    def get_state_for_ai(self) -> dict:
        """Provides the current game state information needed for an AI."""
        # Return piece objects themselves or detailed info if needed by env wrapper
        return {
            "grid_state": self.grid.get_grid_state(), # Get the NumPy array directly
            "current_pieces": self.current_pieces, # Pass the actual Piece objects
            "available_piece_keys": [p.shape_key for p in self.current_pieces], # Keep keys too
            "score": self.score,
            "game_over": self.game_over
        }
    def execute_ai_move(self, piece_index: int, grid_r: int, grid_c: int) -> bool:
        """
        Allows an AI agent to attempt a move.
        Args:
            piece_index: The index (0, 1, or 2) of the piece in self.current_pieces to place.
            grid_r: Target grid row.
            grid_c: Target grid column.
        Returns:
            True if the move was successful, False otherwise.
        """
        if self.game_over or not (0 <= piece_index < len(self.current_pieces)):
            return False

        piece_to_place = self.current_pieces[piece_index]

        # Need to temporarily create a copy or handle the list modification carefully
        # The attempt_placement function removes the piece on success.
        # Let's find the piece by identity or re-fetch based on index if generation happens
        success = self.attempt_placement(piece_to_place, grid_r, grid_c)

        # Note: If placement was successful, self.current_pieces is modified.
        # The AI would typically call get_state_for_ai() *again* after a successful move
        # to see the new state and new pieces (if generated).

        return success