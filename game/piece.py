# piece.py
import pygame
from . import settings as s_game # Use local game settings

class Piece:
    def __init__(self, shape_key, definition):
        self.shape_key = shape_key
        self.relative_cells = definition['shape']
        self.color = definition['color']
        self.screen_pos = (0, 0)
        self.origin_pos = (0, 0)
        self.is_dragging = False

    def set_origin(self, x, y):
        self.origin_pos = (x, y)
        self.screen_pos = (x, y)

    def move_to(self, x, y):
        self.screen_pos = (x, y)

    def get_grid_cells(self, grid_x, grid_y):
        return [(r + grid_y, c + grid_x) for r, c in self.relative_cells]

    def get_block_count(self):
        return len(self.relative_cells)

    def draw(self, surface, ghost=False):
        opacity = 128 if ghost else 255
        for r_off, c_off in self.relative_cells:
            rect = pygame.Rect(
                self.screen_pos[0] + c_off * s_game.TILE_SIZE,
                self.screen_pos[1] + r_off * s_game.TILE_SIZE,
                s_game.TILE_SIZE,
                s_game.TILE_SIZE
            )
            block_surface = pygame.Surface((s_game.TILE_SIZE, s_game.TILE_SIZE), pygame.SRCALPHA)
            block_surface.fill((*self.color, opacity))
            surface.blit(block_surface, rect.topleft)
            pygame.draw.rect(surface, s_game.GRAY, rect, s_game.BLOCK_BORDER_WIDTH)

    def reset_position(self):
        self.screen_pos = self.origin_pos
        self.is_dragging = False

    def get_bounds(self):
        min_r = min(r for r, c in self.relative_cells) if self.relative_cells else 0
        max_r = max(r for r, c in self.relative_cells) if self.relative_cells else 0
        min_c = min(c for r, c in self.relative_cells) if self.relative_cells else 0
        max_c = max(c for r, c in self.relative_cells) if self.relative_cells else 0
        
        width = (max_c - min_c + 1) * s_game.TILE_SIZE
        height = (max_r - min_r + 1) * s_game.TILE_SIZE

        origin_x = self.screen_pos[0] # Assuming screen_pos is top-left of the 0,0 relative cell
        origin_y = self.screen_pos[1]

        # If your piece's relative_cells are not anchored at (0,0) as the top-left most block,
        # you might need to adjust origin_x, origin_y based on min_c, min_r.
        # Assuming (0,0) in relative_cells is the drawing anchor point.
        
        return pygame.Rect(origin_x, origin_y, width, height)