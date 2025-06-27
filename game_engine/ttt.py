import numpy as np
import torch

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 for empty, 1 for X, -1 for O
        self.current_player = 1  # Start with player X
        self.player = 1
        self.turn = 1

    def reset(self):
        new_game = TicTacToe()
        return new_game
        

    def step(self, action):
        new_game = self.copy()
        row, col = action
        if new_game.make_move(row, col):
            new_game.change_player()
            return new_game
        else:
            raise ValueError("Invalid move: Cell already occupied.")

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            return True
        return False
    
    def valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def switch_player(self):
        self.current_player *= -1
        self.player = self.current_player
        self.turn = self.current_player

    def change_player(self):
        self.switch_player()

    def get_feature_plane(self):
        """
        Returns a feature vector representation of the board.
        1 for friend, -1 for enemy, 0 for empty cells.
        """
        return self.current_player * torch.tensor(self.board, dtype=torch.float32).view(1, 3, 3)

    def check_winner(self):
        """
        Return 1 if player X wins, -1 if player O wins, 0 if draw, None if game is still ongoing.
        Checks rows, columns, and diagonals for a win condition.
        """
        # Check rows
        for i in range(3):
            row_sum = np.sum(self.board[i, :])
            if row_sum == 3:
                return 1  # X wins
            elif row_sum == -3:
                return -1  # O wins
        
        # Check columns
        for i in range(3):
            col_sum = np.sum(self.board[:, i])
            if col_sum == 3:
                return 1  # X wins
            elif col_sum == -3:
                return -1  # O wins
        
        # Check main diagonal
        diag_sum = np.sum(np.diag(self.board))
        if diag_sum == 3:
            return 1  # X wins
        elif diag_sum == -3:
            return -1  # O wins
        
        # Check other diagonal
        anti_diag_sum = np.sum(np.diag(np.fliplr(self.board)))
        if anti_diag_sum == 3:
            return 1  # X wins
        elif anti_diag_sum == -3:
            return -1  # O wins
        
        # Check for draw
        if self.is_full():
            return 0  # Draw
        
        # Game still ongoing
        return None
    
    def is_terminal(self):
        return self.check_winner()
    
    def is_game_over(self):
        return self.check_winner()

    def is_full(self):
        return np.all(self.board != 0)
    
    def copy(self):
        new_game = TicTacToe()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.player = self.player
        new_game.turn = self.turn
        return new_game
    
    def __str__(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        return '\n'.join(' | '.join(symbols[self.board[i, j]] for j in range(3)) for i in range(3))

