# grid.py
import pygame
import numpy as np
import settings as s
from typing import List, Tuple, Optional

class Grid:
    def __init__(self):
        self.width = s.GRID_WIDTH
        self.height = s.GRID_HEIGHT
        self.tile_size = s.TILE_SIZE
        # Grid stores the color of the block placed, or None if empty
        self.grid_data: List[List[Optional[Tuple[int, int, int]]]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

    def is_within_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_cell_empty(self, row: int, col: int) -> bool:
        return self.is_within_bounds(row, col) and self.grid_data[row][col] is None

    def can_place_piece(self, piece, grid_r: int, grid_c: int) -> bool:
        """Checks if the piece can be placed at the specified grid location."""
        for r_off, c_off in piece.relative_cells:
            target_r, target_c = grid_r + r_off, grid_c + c_off
            if not self.is_within_bounds(target_r, target_c) or not self.is_cell_empty(target_r, target_c):
                return False
        return True

    def place_piece(self, piece, grid_r: int, grid_c: int):
        """Places the piece onto the grid data."""
        if not self.can_place_piece(piece, grid_r, grid_c):
            # This shouldn't happen if can_place_piece is checked first, but good for safety
            print("Error: Attempted to place piece in invalid location.")
            return

        for r_off, c_off in piece.relative_cells:
            target_r, target_c = grid_r + r_off, grid_c + c_off
            if self.is_within_bounds(target_r, target_c): # Should always be true now
                self.grid_data[target_r][target_c] = piece.color

    def clear_lines_and_squares(self) -> Tuple[int, int, int]:
        """
        Checks for and clears completed rows, columns, and 3x3 squares.
        Returns: Tuple (cleared_rows_count, cleared_cols_count, cleared_squares_count)
        """
        rows_to_clear = []
        cols_to_clear = []
        squares_to_clear = [] # List of top-left (row, col) of the 3x3 square

        # Check rows
        for r in range(self.height):
            if all(self.grid_data[r][c] is not None for c in range(self.width)):
                rows_to_clear.append(r)

        # Check columns
        for c in range(self.width):
            if all(self.grid_data[r][c] is not None for r in range(self.height)):
                cols_to_clear.append(c)

        # Check 3x3 squares
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

        # Clear marked rows, columns, and squares
        cleared_count = 0 # Count unique cells cleared for scoring potentially later

        # Store cells to clear to avoid double-clearing if row/col/square overlap
        cells_to_clear = set()

        for r in rows_to_clear:
            for c in range(self.width):
                cells_to_clear.add((r, c))
        for c in cols_to_clear:
            for r in range(self.height):
                cells_to_clear.add((r, c))
        for r_start, c_start in squares_to_clear:
            for r in range(r_start, r_start + 3):
                for c in range(c_start, c_start + 3):
                    cells_to_clear.add((r, c))

        for r, c in cells_to_clear:
            self.grid_data[r][c] = None
            cleared_count += 1 # Or just use counts below

        return len(rows_to_clear), len(cols_to_clear), len(squares_to_clear)


    def draw(self, surface):
        """Draws the grid lines and the placed blocks."""
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    s.GRID_START_X + c * self.tile_size,
                    s.GRID_START_Y + r * self.tile_size,
                    self.tile_size,
                    self.tile_size
                )
                # Draw cell background/filled block
                if self.grid_data[r][c] is not None:
                    pygame.draw.rect(surface, self.grid_data[r][c], rect)
                    pygame.draw.rect(surface, s.LIGHT_GRAY, rect, s.BLOCK_BORDER_WIDTH) # Border for filled
                else:
                     pygame.draw.rect(surface, s.EMPTY_CELL_COLOR, rect) # Empty cell color

                # Draw grid lines
                pygame.draw.rect(surface, s.GRID_COLOR, rect, s.GRID_LINE_WIDTH)

        # Draw thicker lines for 3x3 square boundaries
        for i in range(1, 3):
             # Vertical lines
             pygame.draw.line(surface, s.GRAY,
                              (s.GRID_START_X + i * 3 * self.tile_size, s.GRID_START_Y),
                              (s.GRID_START_X + i * 3 * self.tile_size, s.GRID_START_Y + self.height * self.tile_size), 2)
             # Horizontal lines
             pygame.draw.line(surface, s.GRAY,
                              (s.GRID_START_X, s.GRID_START_Y + i * 3 * self.tile_size),
                              (s.GRID_START_X + self.width * self.tile_size, s.GRID_START_Y + i * 3 * self.tile_size), 2)

    # --- AI Interaction Method ---
    def get_grid_state(self) -> np.ndarray: # Change return type hint
        """Returns a copy of the grid state suitable for AI (e.g., 0 for empty, 1 for filled)."""
        state = np.zeros((self.height, self.width), dtype=np.float32) # Use float for TF
        for r in range(self.height):
            for c in range(self.width):
                if self.grid_data[r][c] is not None:
                    state[r, c] = 1.0
        # Reshape for CNN input (add channel dimension)
        return np.expand_dims(state, axis=-1)
    