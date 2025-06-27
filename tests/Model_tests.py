import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
import torch

# Add the parent directory to the path so we can import the game engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import TicTacToeCNN

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up a new model instance before each test."""
        self.model = TicTacToeCNN()
        
        # Create a proper mock with methods
        self.state = MagicMock()
        
        # Properly mock methods
        self.state.valid_moves = MagicMock(return_value=[(0, 0), (0, 1), (0, 2)])
        self.state.get_feature_plane = MagicMock(return_value=torch.zeros((1, 3, 3)))
        self.state.is_terminal = MagicMock(return_value=None)
        self.state.step = MagicMock(return_value=self.state)
        self.state.copy = MagicMock(return_value=self.state)
        
        # Set attributes
        self.state.current_player = 1
        self.state.player = 1

    def test_forward(self):
        """Test the forward method of the model."""
        value, policy = self.model.forward(self.state)
        
        # Check if the output value is a float
        self.assertIsInstance(value, float)
        
        # Check if the policy is a dictionary with valid moves
        self.assertIsInstance(policy, dict)
        sum = 0
        for move in self.state.valid_moves():
            self.assertIn(move, policy)
            self.assertGreaterEqual(policy[move], 0)
            sum += policy[move]

        # Check if the policy sums to 1
        self.assertAlmostEqual(sum, 1.0, places=5)
        print(f"Policy: {policy}")

