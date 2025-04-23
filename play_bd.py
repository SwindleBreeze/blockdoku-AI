# main.py
import pygame
import sys
import settings as s
from game.grid import Grid
from game.game_state import GameState
from game.piece import Piece 

def get_grid_coords_from_mouse(mouse_x, mouse_y):
    """Converts mouse screen coordinates to grid cell coordinates."""
    if not (s.GRID_START_X <= mouse_x < s.GRID_START_X + s.GRID_WIDTH * s.TILE_SIZE and
            s.GRID_START_Y <= mouse_y < s.GRID_START_Y + s.GRID_HEIGHT * s.TILE_SIZE):
        return None # Mouse is outside the grid area

    grid_c = (mouse_x - s.GRID_START_X) // s.TILE_SIZE
    grid_r = (mouse_y - s.GRID_START_Y) // s.TILE_SIZE
    return grid_r, grid_c


def main():
    pygame.init()
    screen = pygame.display.set_mode((s.SCREEN_WIDTH, s.SCREEN_HEIGHT))
    pygame.display.set_caption("Blockdoku")
    clock = pygame.time.Clock()

    grid = Grid()
    game = GameState(grid)

    selected_piece: Piece | None = None
    mouse_offset_x = 0
    mouse_offset_y = 0

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game.game_over: # Ignore game input if game over
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    # Check if clicking on an available piece
                    for piece in game.current_pieces:
                        # Use get_bounds which accounts for actual block positions
                        if piece.get_bounds().collidepoint(event.pos):
                            selected_piece = piece
                            selected_piece.is_dragging = True
                            # Calculate offset from piece's top-left visual corner
                            mouse_offset_x = event.pos[0] - selected_piece.screen_pos[0]
                            mouse_offset_y = event.pos[1] - selected_piece.screen_pos[1]
                            break # Stop after finding the first clicked piece

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selected_piece is not None:
                    # Try to place the piece
                    grid_coords = get_grid_coords_from_mouse(event.pos[0] - mouse_offset_x + s.TILE_SIZE // 2,
                                                            event.pos[1] - mouse_offset_y + s.TILE_SIZE // 2)
                                                            # Check center of top-left block

                    placed = False
                    if grid_coords:
                        target_r, target_c = grid_coords
                        # Adjust target based on the piece's anchor (assuming (0,0) is top-left)
                        # The can_place/place functions use top-left grid cell for the piece's (0,0) relative coord
                        if game.attempt_placement(selected_piece, target_r, target_c):
                             placed = True
                        # else: piece remains selected, snaps back below

                    if not placed:
                        # Snap back to original position if placement failed or outside grid
                        selected_piece.reset_position()

                    selected_piece.is_dragging = False
                    selected_piece = None


            elif event.type == pygame.MOUSEMOTION:
                if selected_piece is not None and selected_piece.is_dragging:
                    # Move piece with mouse, considering the initial click offset
                    selected_piece.move_to(event.pos[0] - mouse_offset_x, event.pos[1] - mouse_offset_y)


        # --- Drawing ---
        screen.fill(s.BLACK) # Background

        # Draw Grid
        grid.draw(screen)

        # Draw UI elements (Score, available pieces)
        game.draw_score(screen)
        game.draw_available_pieces(screen) # Draw pieces not being dragged

        # Draw Ghost Piece (Optional visual aid)
        ghost_piece = None
        if selected_piece and selected_piece.is_dragging:
             grid_coords = get_grid_coords_from_mouse(pygame.mouse.get_pos()[0] - mouse_offset_x + s.TILE_SIZE // 2,
                                                     pygame.mouse.get_pos()[1] - mouse_offset_y + s.TILE_SIZE // 2)
             if grid_coords:
                 target_r, target_c = grid_coords
                 if grid.can_place_piece(selected_piece, target_r, target_c):
                    # Create a temporary ghost to draw
                    ghost_pos_x = s.GRID_START_X + target_c * s.TILE_SIZE
                    ghost_pos_y = s.GRID_START_Y + target_r * s.TILE_SIZE
                    # Need a way to draw the piece at a specific location with transparency
                    # Modify Piece.draw or have a dedicated ghost drawing function
                    # For simplicity now, let's just draw the dragged piece normally
                    pass # Placeholder for ghost drawing logic


        # Draw the currently dragged piece last (on top)
        if selected_piece is not None and selected_piece.is_dragging:
            selected_piece.draw(screen)


        # Draw Game Over screen if applicable
        game.draw_game_over(screen)

        pygame.display.flip()
        clock.tick(s.FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()