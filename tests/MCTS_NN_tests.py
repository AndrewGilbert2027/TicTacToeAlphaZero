import unittest
import sys
import os
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MCTS_NN import Deep_Node, MCTS_Deep
from game_engine.ttt import TicTacToe
from Model import TicTacToeCNN

class TestDeepNode(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model that returns predictable values
        self.model = MagicMock()
        self.model.forward.return_value = (0.5, {(0, 0): 0.2, (0, 1): 0.3, (0, 2): 0.5})
        self.model.get_value.return_value = 0.5
        
        # Create a game state
        self.game = MagicMock()
        self.game.is_terminal.return_value = None
        self.game.get_valid_actions.return_value = [(0, 0), (0, 1), (0, 2)]
        self.game.turn.return_value = 1
        self.game.player = 1
        self.game.step.return_value = self.game
        self.game.copy.return_value = self.game


    def test_node_initialization(self):
        """Test node initialization with a mock state and model."""
        node = Deep_Node(self.game, self.model)
        
        # Check if the node is initialized correctly
        self.assertFalse(node.terminal)
        self.assertEqual(len(node.children), 3)  # Should have 3 valid actions
        self.assertEqual(len(node.Q), 3)
        self.assertEqual(len(node.N), 3)
        self.assertEqual(len(node.P), 3)
        self.assertEqual(node.value, 0.5)
        self.assertEqual(node.N_visits, 0)
        self.assertEqual(node.player, 1)
        self.assertEqual(node.P, {(0, 0): 0.2, (0, 1): 0.3, (0, 2): 0.5})

        for action in node.children:
            self.assertIsNone(node.children[action])
            self.assertEqual(node.N[action], 0)
            self.assertEqual(node.T[action], 1)
            
            # Use assertAlmostEqual for floating point comparison
            self.assertAlmostEqual(node.Q[action], 0.0, places=5)

    def test_node_select_action(self):
        """Test action selection using PUCT."""
        node = Deep_Node(self.game, self.model)
        
        # Mock the PUCT values for actions
        node.P = {(0, 0): 0.0, (0, 1): 0.3, (0, 2): 0.5}
        node.N = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
        node.N_visits = 3
        
        # Select action with PUCT
        selected_action = node._select_action(puct=1.0)
        
        # Check if the action with the highest PUCT value is selected
        self.assertEqual(selected_action, (0, 2))

    def test_node_select_action_terminal(self):
        """Test action selection when node is terminal."""
        node = Deep_Node(self.game, self.model)
        node.terminal = True
        
        # Attempt to select an action
        selected_action = node._select_action(puct=1.0)
        
        # Should return None since the node is terminal
        self.assertIsNone(selected_action)

    def test_node_select_action_no_valid_actions(self):
        """Test action selection when there are no valid actions."""
        node = Deep_Node(self.game, self.model)
        node.P = {}
        
        # Attempt to select an action
        selected_action = node._select_action(puct=1.0)
        
        # Should return None since there are no valid actions
        self.assertIsNone(selected_action)

    def test_node_select_action_simple(self):
        """Test action selection with a simple case."""
        node = Deep_Node(self.game, self.model)
        
        # Mock the PUCT values for actions
        node.P = {(0, 0): 0.1, (0, 1): 0.2, (0, 2): 0.7}
        node.N = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
        node.T = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
        node.N_visits = 3
        
        # Select action with PUCT
        selected_action = node._select_action(puct=1.0)
        
        # Check if the action with the highest PUCT value is selected
        self.assertEqual(selected_action, (0, 2))

    def test_node_select_action_differing_player(self):
        """Test action selection with a differing player."""
        node = Deep_Node(self.game, self.model)
        
        # Mock the PUCT values for actions
        node.P = {(0, 0): 0.1, (0, 1): 0.2, (0, 2): 0.7}
        node.N = {(0, 0): 1, (0, 1): 1, (0, 2): 1}
        node.T = {(0, 0): -1, (0, 1): -1, (0, 2): -1}
        node.N_visits = 3

        selected_action = node._select_action(puct=1.0)

        # Check that we choose (0, 2) since we want to explore
        self.assertEqual(selected_action, (0, 2))

        node.T = {(0, 0): -1, (0, 1): 1, (0, 2): -1}  # Reset 
        node.Q = {(0, 0): 0.1, (0, 1): 1.0, (0, 2): 0.3}
        selected_action = node._select_action(puct=1.0)
        self.assertEqual(selected_action, (0, 1))  # Should choose (0, 1) since it has a Q value now


    def test_mcts_deep_initialization(self):
        """Test MCTS_Deep initialization."""
        mcts = MCTS_Deep(self.game, self.model)

        # Check if the MCTS_Deep object is initialized correctly
        self.assertEqual(mcts.state, self.game)
        self.assertEqual(mcts.model, self.model)
        self.assertIsInstance(mcts.root, Deep_Node)
        

    def test_mcts_select(self):
        """Test MCTS_Deep select method."""
        mcts = MCTS_Deep(self.game, self.model)
        
        # Mock the select method to return a specific action
        mcts.root._select_action = MagicMock(return_value=(0, 0))
        
        selected_action = mcts._select()
        
        # Check if the selected action matches the mocked value
        self.assertEqual(selected_action, [(mcts.root, (0, 0))])

        mcts = MCTS_Deep(self.game, self.model)
        selected_action = mcts._select()
        self.assertEqual(selected_action, [(mcts.root, (0,0))])  # Should return None if no valid actions


    def test_mcts_expand(self):
        """Test MCTS_Deep expand method."""
        mcts = MCTS_Deep(self.game, self.model)
        
        # Mock the expand method to return a new node
        mcts.root._expand = MagicMock(return_value=Deep_Node(self.game, self.model))
        
        path = [(mcts.root, (0, 0))]
        expanded_node = mcts._expand(path)
        
        # Check if the expanded node is an instance of Deep_Node
        self.assertIsInstance(expanded_node, Deep_Node)

    def test_mcts_backup(self):
        """Test MCTS_Deep backup method."""
        mcts = MCTS_Deep(self.game, self.model)
        
        # Setup the root node with proper T dictionary
        mcts.root.T[(0, 0)] = 1  # Initialize T for the action we'll use
        
        # Create a child node and add it to the root's children
        child_node = MagicMock()
        child_node.value = 0.1
        child_node.player = mcts.root.player  # Same player for simplicity
        mcts.root.children[(0, 0)] = child_node
        
        # Create path and run backup
        path = [(mcts.root, (0, 0))]
        value = 0.1
        mcts._backup(path, value)
        
        # Check if the update worked correctly
        self.assertEqual(mcts.root.N_visits, 1)
        self.assertEqual(mcts.root.N[(0, 0)], 1)
        self.assertEqual(mcts.root.Q[(0, 0)], 0.1)  # Q should be updated to child's value
        self.assertAlmostEqual(mcts.root.value, 0.1, places=5)


    def test_mcts_search_with_tictactoe(self):
        game = TicTacToe()
        model = self.model
        mcts = MCTS_Deep(game, model)

        self.assertEqual(mcts.root.P, {(0, 0): 0.2, (0, 1): 0.3, (0, 2): 0.5})
        self.assertEqual(mcts.root.N, {(0, 0): 0, (0, 1): 0, (0, 2): 0})
        self.assertEqual(mcts.root.Q, {(0, 0): 0.0, (0, 1): 0.0, (0, 2): 0.0})
        self.assertEqual(mcts.root.value, 0.5)
        self.assertEqual(mcts.root.N_visits, 0)
        self.assertEqual(mcts.root.player, 1)
        for action in mcts.root.P:
            self.assertEqual(mcts.root.children[action], None)
            self.assertEqual(mcts.root.T[action], -1) # T should switch value

















