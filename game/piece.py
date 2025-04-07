# piece.py
import pygame
import settings as s

class Piece:
    def __init__(self, shape_key, definition):
        self.shape_key = shape_key # e.g., 'l_shape'
        self.relative_cells = definition['shape'] # List of (row, col) offsets
        self.color = definition['color']
        self.screen_pos = (0, 0) # Top-left corner for drawing purposes
        self.origin_pos = (0, 0) # Where it sits in the selection area
        self.is_dragging = False

    def set_origin(self, x, y):
        """Sets the initial position in the selection area."""
        self.origin_pos = (x, y)
        self.screen_pos = (x, y)

    def move_to(self, x, y):
        """Moves the top-left drawing position."""
        self.screen_pos = (x, y)

    def get_grid_cells(self, grid_x, grid_y):
        """Calculates the absolute grid cells occupied if placed at grid_x, grid_y."""
        return [(r + grid_y, c + grid_x) for r, c in self.relative_cells]

    def get_block_count(self):
        return len(self.relative_cells)

    def draw(self, surface, ghost=False):
        """Draws the piece on the given surface."""
        opacity = 128 if ghost else 255 # Semi-transparent for ghost
        for r_off, c_off in self.relative_cells:
            rect = pygame.Rect(
                self.screen_pos[0] + c_off * s.TILE_SIZE,
                self.screen_pos[1] + r_off * s.TILE_SIZE,
                s.TILE_SIZE,
                s.TILE_SIZE
            )
            # Create a surface with per-pixel alpha for transparency
            block_surface = pygame.Surface((s.TILE_SIZE, s.TILE_SIZE), pygame.SRCALPHA)
            block_surface.fill((*self.color, opacity)) # Use RGBA

            surface.blit(block_surface, rect.topleft)

            # Draw border around each block
            pygame.draw.rect(surface, s.GRAY, rect, s.BLOCK_BORDER_WIDTH)

    def reset_position(self):
        """Snaps the piece back to its origin position."""
        self.screen_pos = self.origin_pos
        self.is_dragging = False

    def get_bounds(self):
        """Returns the bounding box Rect in screen coordinates."""
        min_r = min(r for r, c in self.relative_cells)
        max_r = max(r for r, c in self.relative_cells)
        min_c = min(c for r, c in self.relative_cells)
        max_c = max(c for r, c in self.relative_cells)
        width = (max_c - min_c + 1) * s.TILE_SIZE
        height = (max_r - min_r + 1) * s.TILE_SIZE

        # Adjust position based on the minimum row/col offsets
        # This assumes screen_pos refers to the (0,0) relative cell
        origin_x = self.screen_pos[0] + min_c * s.TILE_SIZE
        origin_y = self.screen_pos[1] + min_r * s.TILE_SIZE

        return pygame.Rect(origin_x, origin_y, width, height)