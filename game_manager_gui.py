import pygame
import sys
import random
from game_engine.ttt import TicTacToe
from MCTS import MCTS
from MCTS_NN import Deep_Node, MCTS_Deep
from Model import TicTacToeCNN
import torch

class TicTacToeGUI:
    # Colors
    COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'gray': (200, 200, 200),
    }
    
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Constants
        self.WIDTH, self.HEIGHT = 600, 700  # Extra height for status display
        self.LINE_WIDTH = 15
        self.BOARD_SIZE = 3
        self.CELL_SIZE = self.WIDTH // self.BOARD_SIZE
        self.BG_COLOR = self.COLORS['white']
        
        # Set up screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe")
        
        # Game state
        self.game = TicTacToe()
        self.game_over = False
        self.reset_button = None
        
        # Fixed colors for X and O
        self.x_color = self.COLORS['blue']
        self.o_color = self.COLORS['red']
        
        # Randomly assign X and O to AI and human players
        self.randomize_players()

        self.ai_model = TicTacToeCNN()
        self.ai_model.load_state_dict(torch.load('models/b_model_epoch_12.pth', map_location=torch.device('cpu')))
    
    def randomize_players(self):
        """Randomly assign X and O to AI and human players"""
        players = [1, -1]
        random.shuffle(players)
        self.ai_player, self.human_player = players[0], players[1]
        
        # Display who's playing as what
        player_roles = "AI plays as " + ("X" if self.ai_player == 1 else "O") + ", Human plays as " + ("X" if self.human_player == 1 else "O")
        print(player_roles)
    
    def draw_board(self):
        """Draw the Tic-Tac-Toe grid"""
        # Horizontal lines
        pygame.draw.line(self.screen, self.COLORS['black'], 
                        (0, self.CELL_SIZE), 
                        (self.WIDTH, self.CELL_SIZE), 
                        self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.COLORS['black'], 
                        (0, 2 * self.CELL_SIZE), 
                        (self.WIDTH, 2 * self.CELL_SIZE), 
                        self.LINE_WIDTH)
        
        # Vertical lines
        pygame.draw.line(self.screen, self.COLORS['black'], 
                        (self.CELL_SIZE, 0), 
                        (self.CELL_SIZE, self.WIDTH), 
                        self.LINE_WIDTH)
        pygame.draw.line(self.screen, self.COLORS['black'], 
                        (2 * self.CELL_SIZE, 0), 
                        (2 * self.CELL_SIZE, self.WIDTH), 
                        self.LINE_WIDTH)

    def draw_figures(self):
        """Draw X and O based on the game board state"""
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                cell_value = self.game.board[row, col]
                center_x = col * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = row * self.CELL_SIZE + self.CELL_SIZE // 2
                
                if cell_value == 1:  # X
                    # Draw X
                    radius = self.CELL_SIZE // 3
                    pygame.draw.line(self.screen, self.x_color, 
                                    (center_x - radius, center_y - radius),
                                    (center_x + radius, center_y + radius), 
                                    self.LINE_WIDTH)
                    pygame.draw.line(self.screen, self.x_color, 
                                    (center_x + radius, center_y - radius),
                                    (center_x - radius, center_y + radius), 
                                    self.LINE_WIDTH)
                elif cell_value == -1:  # O
                    # Draw O
                    radius = self.CELL_SIZE // 3
                    pygame.draw.circle(self.screen, self.o_color, 
                                      (center_x, center_y), radius, self.LINE_WIDTH)

    def draw_status(self):
        """Draw game status and return reset button if game over"""
        status_rect = pygame.Rect(0, self.WIDTH, self.WIDTH, self.HEIGHT - self.WIDTH)
        pygame.draw.rect(self.screen, self.COLORS['gray'], status_rect)
        
        font = pygame.font.SysFont(None, 30)
        
        # Display which player is AI and which is human
        player_info = "AI: " + ("X" if self.ai_player == 1 else "O") + " | Human: " + ("X" if self.human_player == 1 else "O")
        player_text = font.render(player_info, True, self.COLORS['black'])
        player_rect = player_text.get_rect(midtop=(self.WIDTH // 2, self.WIDTH + 10))
        self.screen.blit(player_text, player_rect)
        
        if self.game_over:
            winner = self.game.check_winner()
            if winner == 1:
                status_text = "X wins!"
            elif winner == -1:
                status_text = "O wins!"
            else:
                status_text = "It's a draw!"
        else:
            if self.game.current_player == 1:
                status_text = "Player X's turn"
            else:
                status_text = "Player O's turn"
        
        text_surface = font.render(status_text, True, self.COLORS['black'])
        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, 
                                                 self.WIDTH + (self.HEIGHT - self.WIDTH) // 2))
        self.screen.blit(text_surface, text_rect)
        
        # Add reset button when game is over
        if self.game_over:
            button_rect = pygame.Rect(self.WIDTH // 4, 
                                     self.WIDTH + (self.HEIGHT - self.WIDTH) * 3 // 4, 
                                     self.WIDTH // 2, 40)
            pygame.draw.rect(self.screen, self.COLORS['black'], button_rect, 2)
            
            reset_text = font.render("Play Again", True, self.COLORS['black'])
            reset_text_rect = reset_text.get_rect(center=button_rect.center)
            self.screen.blit(reset_text, reset_text_rect)
            
            return button_rect
        return None

    def check_game_over(self):
        """Check if the game is over (win or draw)"""
        return self.game.check_winner() is not None

    def reset_game(self):
        """Reset the game state and randomize players"""
        self.game = self.game.reset()
        self.game_over = False
        self.randomize_players()

    def handle_ai_turn(self, deep=False):
        """Handle AI's turn using MCTS"""
        if self.game.current_player == self.ai_player and not self.game_over:
            if deep == False:
                mcts = MCTS(self.game, max_iterations=2000)
                best_move = mcts.search()
                if best_move:
                    self.game.make_move(best_move[0], best_move[1])
                    if self.check_game_over():
                        self.game_over = True
                    else:
                        self.game.switch_player()
            else:
                mcts_deep = MCTS_Deep(self.game, self.ai_model)
                val, visits = mcts_deep.search(num_simulations=1500, puct=1.0)
                print(f"AI Value: {val}, Visits: {visits}")
                best_move = max(visits.items(), key=lambda x: x[1])[0] if visits else None
                if best_move:
                    self.game.make_move(best_move[0], best_move[1])
                    if self.check_game_over():
                        self.game_over = True
                    else:
                        self.game.switch_player()

    def handle_mouse_click(self, pos):
        """Handle mouse click events"""
        mouseX, mouseY = pos
        
        # Handle game board clicks
        if not self.game_over and self.game.current_player == self.human_player:
            # Convert to board coordinates
            clicked_row = mouseY // self.CELL_SIZE
            clicked_col = mouseX // self.CELL_SIZE
            
            # Ensure click is within board bounds
            if 0 <= clicked_row < self.BOARD_SIZE and 0 <= clicked_col < self.BOARD_SIZE:
                # Make move if valid
                if self.game.make_move(clicked_row, clicked_col):
                    # Check if game is over after move
                    if self.check_game_over():
                        self.game_over = True
                    else:
                        # Switch player for next turn
                        self.game.switch_player()
        
        # Handle reset button click
        elif self.game_over and self.reset_button and self.reset_button.collidepoint(mouseX, mouseY):
            self.reset_game()

    def run(self, deep=False):
        """Main game loop"""
        running = True
        while running:
            # Handle AI turn
            self.handle_ai_turn(deep=deep)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event.pos)
            
            # Update display
            self.screen.fill(self.BG_COLOR)
            self.draw_board()
            self.draw_figures()
            self.reset_button = self.draw_status()
            pygame.display.update()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game_gui = TicTacToeGUI()
    game_gui.run(deep=True)  # Set to True for deep MCTS with neural network)
