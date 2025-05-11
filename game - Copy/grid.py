# grid.py
import pygame
import numpy as np
from . import settings as s_game # Use local game settings
from typing import List, Tuple, Optional
# Piece class might be needed if type hinting 'piece' arguments, but not strictly for settings
# from .piece import Piece 

class Grid:
    def __init__(self):
        self.width = s_game.GRID_WIDTH
        self.height = s_game.GRID_HEIGHT
        self.tile_size = s_game.TILE_SIZE
        self.grid_data: List[List[Optional[Tuple[int, int, int]]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

    def is_within_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_cell_empty(self, row: int, col: int) -> bool:
        return self.is_within_bounds(row, col) and self.grid_data[row][col] is None

    def can_place_piece(self, piece, grid_r: int, grid_c: int) -> bool: # piece type is Piece
        for r_off, c_off in piece.relative_cells:
            target_r, target_c = grid_r + r_off, grid_c + c_off
            if not self.is_within_bounds(target_r, target_c) or not self.is_cell_empty(target_r, target_c):
                return False
        return True

    def place_piece(self, piece, grid_r: int, grid_c: int): # piece type is Piece
        if not self.can_place_piece(piece, grid_r, grid_c):
            print("Error: Attempted to place piece in invalid location.")
            return

        for r_off, c_off in piece.relative_cells:
            target_r, target_c = grid_r + r_off, grid_c + c_off
            if self.is_within_bounds(target_r, target_c):
                self.grid_data[target_r][target_c] = piece.color

    def clear_lines_and_squares(self) -> Tuple[int, int, int]:
        rows_to_clear = []
        cols_to_clear = []
        squares_to_clear = [] 

        for r in range(self.height):
            if all(self.grid_data[r][c] is not None for c in range(self.width)):
                rows_to_clear.append(r)

        for c in range(self.width):
            if all(self.grid_data[r][c] is not None for r in range(self.height)):
                cols_to_clear.append(c)

        for r_start in range(0, self.height, 3):
            for c_start in range(0, self.width, 3):
                is_full = True
                for r in range(r_start, r_start + 3):
                    for c in range(c_start, c_start + 3):
                        if self.grid_data[r][c] is None:
                            is_full = False
                            break
                    if not is_full:
                        break
                if is_full:
                    squares_to_clear.append((r_start, c_start))

        cells_to_clear = set()
        for r_idx in rows_to_clear:
            for c in range(self.width):
                cells_to_clear.add((r_idx, c))
        for c_idx in cols_to_clear:
            for r in range(self.height):
                cells_to_clear.add((r, c_idx))
        for r_start, c_start in squares_to_clear:
            for r in range(r_start, r_start + 3):
                for c in range(c_start, c_start + 3):
                    cells_to_clear.add((r, c))

        for r, c in cells_to_clear:
            self.grid_data[r][c] = None
            
        return len(rows_to_clear), len(cols_to_clear), len(squares_to_clear)


    def draw(self, surface):
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    s_game.GRID_START_X + c * self.tile_size,
                    s_game.GRID_START_Y + r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                if self.grid_data[r][c] is not None:
                    pygame.draw.rect(surface, self.grid_data[r][c], rect)
                    pygame.draw.rect(surface, s_game.LIGHT_GRAY, rect, s_game.BLOCK_BORDER_WIDTH)
                else:
                     pygame.draw.rect(surface, s_game.EMPTY_CELL_COLOR, rect)
                pygame.draw.rect(surface, s_game.GRID_COLOR, rect, s_game.GRID_LINE_WIDTH)

        for i in range(1, self.width // 3): # Adjusted for 9x9 grid, gives 2 lines
             pygame.draw.line(surface, s_game.GRAY,
                              (s_game.GRID_START_X + i * 3 * self.tile_size, s_game.GRID_START_Y),
                              (s_game.GRID_START_X + i * 3 * self.tile_size, s_game.GRID_START_Y + self.height * self.tile_size), 2)
             pygame.draw.line(surface, s_game.GRAY,
                              (s_game.GRID_START_X, s_game.GRID_START_Y + i * 3 * self.tile_size),
                              (s_game.GRID_START_X + self.width * self.tile_size, s_game.GRID_START_Y + i * 3 * self.tile_size), 2)

    def get_grid_state(self) -> np.ndarray:
        state = np.zeros((self.height, self.width), dtype=np.float32)
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_data[r][c] is not None:
                    state[r, c] = 1.0
        return np.expand_dims(state, axis=-1) # Adds channel dim, uses s_game.STATE_GRID_CHANNELS implicitly (1)
    
    def count_filled_in_row(self, r_idx: int) -> int:
        if not (0 <= r_idx < self.height):
            return 0
        return sum(1 for c in range(self.width) if self.grid_data[r_idx][c] is not None)

    def count_filled_in_col(self, c_idx: int) -> int:
        if not (0 <= c_idx < self.width):
            return 0
        return sum(1 for r in range(self.height) if self.grid_data[r][c_idx] is not None)

    def count_filled_in_square(self, r_start: int, c_start: int) -> int:
        count = 0
        if not (0 <= r_start <= self.height - 3 and 0 <= c_start <= self.width - 3):
            return 0
        for r in range(r_start, r_start + 3):
            for c in range(c_start, c_start + 3):
                if self.grid_data[r][c] is not None:
                    count += 1
        return count

    def get_almost_full_regions_info(self, piece_cells_just_placed: set) -> Tuple[int, List[str]]:
        """
        Checks rows, cols, and squares affected by the recently placed piece
        to see if they are now "almost full" (7 or 8 out of 9).
        Returns a count of newly "almost full" regions and their types.
        """
        newly_almost_full_count = 0
        almost_full_details = [] # For debugging: e.g., ["row_3", "col_5_sq_0_0"]

        affected_rows = set()
        affected_cols = set()
        affected_squares = set() # Store as (r_start, c_start)

        for r, c in piece_cells_just_placed:
            affected_rows.add(r)
            affected_cols.add(c)
            affected_squares.add(( (r // 3) * 3, (c // 3) * 3 ))

        # Check affected rows
        for r_idx in affected_rows:
            filled_count = self.count_filled_in_row(r_idx)
            if filled_count == 7 or filled_count == 8: # Almost full but not completely full
                newly_almost_full_count += 1
                almost_full_details.append(f"row_{r_idx}")
        
        # Check affected columns
        for c_idx in affected_cols:
            filled_count = self.count_filled_in_col(c_idx)
            if filled_count == 7 or filled_count == 8:
                # Avoid double counting if a cell completes both an almost full row and col in terms of regions
                # This counts regions, so if a piece makes a row almost full and a col almost full, that's 2 events.
                newly_almost_full_count += 1
                almost_full_details.append(f"col_{c_idx}")

        # Check affected 3x3 squares
        for r_start, c_start in affected_squares:
            filled_count = self.count_filled_in_square(r_start, c_start)
            if filled_count == 7 or filled_count == 8:
                newly_almost_full_count += 1
                almost_full_details.append(f"sq_{r_start}_{c_start}")
        
        # This simple counting might over-reward if a single piece placement
        # makes multiple regions "almost full" simultaneously and they were already close.
        # A more refined approach would check the state *before* placing the piece's last block
        # for that specific region, but that's much more complex.
        # For now, this counts how many relevant regions are now in an "almost full" state.
        # We assume this reward is for the *state achieved*, not the delta.
        # To refine, we'd need to know the *previous* count of almost_full_regions.
        # For simplicity, let's just count current "almost full" regions touched by the new piece.

        return newly_almost_full_count, almost_full_details


    def clear_lines_and_squares(self) -> Tuple[int, int, int]:
        # ... (implementation remains the same)
        rows_to_clear = []
        cols_to_clear = []
        squares_to_clear = [] 

        for r in range(self.height):
            if self.count_filled_in_row(r) == self.width: # A full row has 9 cells
                rows_to_clear.append(r)

        for c in range(self.width):
            if self.count_filled_in_col(c) == self.height: # A full col has 9 cells
                cols_to_clear.append(c)

        for r_start in range(0, self.height -2): # Iterate 0, 3, 6 for a 9x9 grid
            for c_start in range(0, self.width -2): # Iterate 0, 3, 6
                if (r_start % 3 == 0 and c_start % 3 == 0): # Ensure it's a valid 3x3 square start
                    if self.count_filled_in_square(r_start, c_start) == 9: # A full 3x3 square
                        squares_to_clear.append((r_start, c_start))
        
        # Remove duplicates if a square clear also implies row/col clears (e.g. big_square piece)
        # The current clearing logic handles cells_to_clear set correctly.
        # The counts returned are distinct region clear events.

        cells_to_clear = set()
        for r_idx in rows_to_clear:
            for c in range(self.width):
                cells_to_clear.add((r_idx, c))
        for c_idx in cols_to_clear:
            for r in range(self.height):
                cells_to_clear.add((r, c_idx))
        for r_start, c_start in squares_to_clear:
            for r in range(r_start, r_start + 3):
                for c in range(c_start, c_start + 3):
                    cells_to_clear.add((r, c))

        for r_val, c_val in cells_to_clear: # Renamed to avoid conflict
            self.grid_data[r_val][c_val] = None
            
        return len(rows_to_clear), len(cols_to_clear), len(squares_to_clear)