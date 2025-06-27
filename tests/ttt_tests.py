import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the game engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_engine.ttt import TicTacToe

class TestTicTacToe(unittest.TestCase):
    
    def setUp(self):
        """Set up a new game before each test."""
        self.game = TicTacToe()
    
    def test_initialization(self):
        """Test that the game initializes correctly."""
        self.assertTrue(np.array_equal(self.game.board, np.zeros((3, 3))))
        self.assertEqual(self.game.current_player, 1)
    
    def test_make_move(self):
        """Test making valid and invalid moves."""
        # Test valid move
        self.assertTrue(self.game.make_move(0, 0))
        self.assertEqual(self.game.board[0, 0], 1)
        
        # Test invalid move (already occupied)
        self.assertFalse(self.game.make_move(0, 0))
    
    def test_switch_player(self):
        """Test that players switch correctly."""
        self.assertEqual(self.game.current_player, 1)
        self.game.switch_player()
        self.assertEqual(self.game.current_player, -1)
        self.game.switch_player()
        self.assertEqual(self.game.current_player, 1)
    
    def test_row_win(self):
        """Test winning by completing a row."""
        # Player 1 makes a row
        self.game.make_move(0, 0)
        self.game.make_move(0, 1)
        self.game.make_move(0, 2)
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and test another row
        self.game.reset()
        self.game.make_move(1, 0)
        self.game.make_move(1, 1)
        self.game.make_move(1, 2)
        self.assertEqual(self.game.check_winner(), 1)
    
    def test_column_win(self):
        """Test winning by completing a column."""
        # Player 1 makes a column
        self.game.make_move(0, 0)
        self.game.make_move(1, 0)
        self.game.make_move(2, 0)
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and test another column
        self.game.reset()
        self.game.make_move(0, 2)
        self.game.make_move(1, 2)
        self.game.make_move(2, 2)
        self.assertEqual(self.game.check_winner(), 1)
    
    def test_diagonal_win(self):
        """Test winning by completing a diagonal."""
        # Player 1 makes the main diagonal
        self.game.make_move(0, 0)
        self.game.make_move(1, 1)
        self.game.make_move(2, 2)
        self.assertEqual(self.game.check_winner(), 1)
        
        # Reset and test the other diagonal
        self.game.reset()
        self.game.make_move(0, 2)
        self.game.make_move(1, 1)
        self.game.make_move(2, 0)
        self.assertEqual(self.game.check_winner(), 1)
    
    def test_player2_win(self):
        """Test that player 2 can win."""
        # Setup player 2's turn
        self.game.make_move(0, 0)  # Player 1
        self.game.switch_player()
        
        # Player 2 makes a row
        self.game.make_move(1, 0)
        self.game.make_move(1, 1)
        self.game.make_move(1, 2)
        self.assertEqual(self.game.check_winner(), -1)
    
    def test_draw(self):
        """Test that the game correctly identifies a draw."""
        # Fill the board without a winner
        moves = [
            (0, 0), (0, 1), (0, 2),
            (1, 1), (1, 0), (1, 2),
            (2, 1), (2, 0), (2, 2)
        ]
        
        for i, (row, col) in enumerate(moves):
            self.game.make_move(row, col)
            self.game.switch_player()
            if i < 8:  # Not full yet
                self.assertIsNone(self.game.check_winner())
        
        # Board is now full without a winner
        self.assertTrue(self.game.is_full())
        self.assertEqual(self.game.check_winner(), 0)
    
    def test_reset(self):
        """Test that reset clears the board and resets the player."""
        # Make some moves
        self.game.make_move(0, 0)
        self.game.switch_player()
        self.game.make_move(1, 1)
        
        # Reset and check
        new_game = self.game.reset()
        self.assertTrue(np.array_equal(new_game.board, np.zeros((3, 3))))
        self.assertEqual(new_game.current_player, 1)

    def test_get_game_plane_initial(self):
        plane = self.game.get_feature_plane()
        self.assertEqual(plane.shape, (1, 3, 3))
        self.assertTrue(np.array_equal(plane, np.zeros((1, 3, 3))))

    def test_get_game_plane_after_moves(self):
        self.game.make_move(0, 0) # player 1 in top left
        self.game.make_move(1, 1) # player 1 in middle (did not change player)
        plane = self.game.get_feature_plane()
        self.assertEqual(plane.shape, (1, 3, 3))
        expected_plane = np.array([[[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]], dtype=np.float32)
        self.assertTrue(np.array_equal(plane.numpy(), expected_plane))

    def test_get_game_plane_after_moves_and_player_switch(self):
        self.game.make_move(0, 0)
        self.game.switch_player()
        self.game.make_move(1, 1)
        plane = self.game.get_feature_plane()
        self.assertEqual(plane.shape, (1, 3, 3))
        expected_plane = np.array([[[-1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]], dtype=np.float32)
        self.assertTrue(np.array_equal(plane.numpy(), expected_plane))

    def test_game_plane_after_step(self):
        new_game = self.game.step((0, 0))
        plane = new_game.get_feature_plane()
        self.assertEqual(plane.shape, (1, 3, 3))
        expected_plane = np.array([[[-1, 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]]], dtype=np.float32)

if __name__ == '__main__':
    unittest.main()
