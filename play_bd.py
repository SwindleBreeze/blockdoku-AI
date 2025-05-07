# play_bd.py
import pygame
import sys
# Explicitly import game settings for game visuals and mechanics
from game import settings as s_game 
from game.grid import Grid
from game.game_state import GameState
# Piece is used but its settings come from s_game indirectly through PIECE_DEFINITIONS
# from game.piece import Piece 

def get_grid_coords_from_mouse(mouse_x, mouse_y):
    """Converts mouse screen coordinates to grid cell coordinates."""
    if not (s_game.GRID_START_X <= mouse_x < s_game.GRID_START_X + s_game.GRID_WIDTH * s_game.TILE_SIZE and
            s_game.GRID_START_Y <= mouse_y < s_game.GRID_START_Y + s_game.GRID_HEIGHT * s_game.TILE_SIZE):
        return None 

    grid_c = (mouse_x - s_game.GRID_START_X) // s_game.TILE_SIZE
    grid_r = (mouse_y - s_game.GRID_START_Y) // s_game.TILE_SIZE
    return grid_r, grid_c


def main():
    pygame.init()
    pygame.font.init() # Initialize font module
    screen = pygame.display.set_mode((s_game.SCREEN_WIDTH, s_game.SCREEN_HEIGHT))
    pygame.display.set_caption("Blockdoku")
    clock = pygame.time.Clock()

    grid = Grid()
    game = GameState(grid)

    selected_piece = None # Explicitly type hint if possible: Optional[Piece] = None
    mouse_offset_x = 0
    mouse_offset_y = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game.game_over: 
                continue

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    for piece in game.current_pieces:
                        if piece.get_bounds().collidepoint(event.pos):
                            selected_piece = piece
                            selected_piece.is_dragging = True
                            mouse_offset_x = event.pos[0] - selected_piece.screen_pos[0]
                            mouse_offset_y = event.pos[1] - selected_piece.screen_pos[1]
                            break 

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and selected_piece is not None:
                    grid_coords = get_grid_coords_from_mouse(
                        event.pos[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                        event.pos[1] - mouse_offset_y + s_game.TILE_SIZE // 2
                    )
                    
                    placed = False
                    if grid_coords:
                        target_r, target_c = grid_coords
                        # attempt_placement now returns success, r_cleared, c_cleared, sq_cleared
                        success_place, _, _, _ = game.attempt_placement(selected_piece, target_r, target_c)
                        if success_place:
                             placed = True
                    
                    if not placed:
                        selected_piece.reset_position()

                    selected_piece.is_dragging = False
                    selected_piece = None

            elif event.type == pygame.MOUSEMOTION:
                if selected_piece is not None and selected_piece.is_dragging:
                    selected_piece.move_to(event.pos[0] - mouse_offset_x, event.pos[1] - mouse_offset_y)

        screen.fill(s_game.BLACK)
        grid.draw(screen)
        game.draw_score(screen)
        game.draw_available_pieces(screen)

        # Ghost piece logic (optional, can be enhanced)
        if selected_piece and selected_piece.is_dragging:
             grid_coords_ghost = get_grid_coords_from_mouse(
                 pygame.mouse.get_pos()[0] - mouse_offset_x + s_game.TILE_SIZE // 2,
                 pygame.mouse.get_pos()[1] - mouse_offset_y + s_game.TILE_SIZE // 2
             )
             if grid_coords_ghost:
                 target_r_g, target_c_g = grid_coords_ghost
                 if grid.can_place_piece(selected_piece, target_r_g, target_c_g):
                    original_screen_pos = selected_piece.screen_pos # Store original
                    selected_piece.screen_pos = (
                        s_game.GRID_START_X + target_c_g * s_game.TILE_SIZE,
                        s_game.GRID_START_Y + target_r_g * s_game.TILE_SIZE
                    )
                    selected_piece.draw(screen, ghost=True) # Assuming Piece.draw handles ghost
                    selected_piece.screen_pos = original_screen_pos # Restore

        if selected_piece is not None and selected_piece.is_dragging:
            selected_piece.draw(screen) # Draw the actual dragged piece

        game.draw_game_over(screen)

        pygame.display.flip()
        clock.tick(s_game.FPS)

    pygame.font.quit()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()