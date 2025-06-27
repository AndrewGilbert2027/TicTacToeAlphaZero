"""
This class contains the implementation of the Monte Carlo Tree Search (MCTS) algorithm. 
It is designed to be used with a game environment that provides the necessary methods for state management, action generation, and reward evaluation.
"""
import numpy as np

class MCTS_NODE:
    def __init__(self):
        self.state = None
        self.parent = None
        self.children = {} # Dictionary to hold child nodes with actions as keys
        self.visits = 0
        self.value_sum = 0
        self.untried_actions = []
        self.player = None
    
class MCTS:
    def __init__(self, game, max_iterations=10):
        self.game = game # The game environment (Holds the game state and methods)
        self.max_iterations = max_iterations
        self.root = MCTS_NODE()
        self.root.state = self.game.copy() # Copy the initial game state
        self.root.untried_actions = self.game.valid_moves()  # Initial valid moves
        self.root.player = self.game.current_player  # Current player
        self.root.visits = 0  # Root node starts with zero visits
        self.root.value_sum = 0  # Root node starts with zero value_sum

    def search(self, max_iterations=None):
        """
        Perform the MCTS search.
        :param max_iterations: Maximum number of iterations to run the search.
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        for _ in range(max_iterations):
            node = self._select(self.root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        return self._best_action(self.root)
    
    def _select(self, node):
        """
        Select a node to expand.
        :param node: The current node to select from.
        :return: The selected node.
        """
        # Check if the node represents a terminal state
        if node.state.is_game_over() is not None:
            return node
        
        # if untried actions are available, select a random untried action
        if node.untried_actions:
            action = node.untried_actions.pop() # select and remove an action 
            child_node = self._expand(node, action) # expand the node with the selected action (create a new child node)
            return child_node
        else:
            # Choose a node based off the UCT value (Upper Confidence Bound for Trees) 
            best_child = max(node.children.items(), key=lambda item: self._uct_value(item[1]))[1]
            return self._select(best_child) # recursively select from the best child node
        
    def _best_action(self, node):
        """
        Get the best action from the root node.
        :param node: The root node to get the best action from.
        :return: The best action based on the player.
        """
        if not node.children:
            return None
        
        # Choose the action with the maximum visits
        best_action = max(node.children.items(), key=lambda item: item[1].visits)[0]
        return best_action
        

    def _expand(self, node, action):
        """
        Expand the node by adding a child node for the given action.
        :param node: The node to expand.
        :param action: The action to expand with.
        :return: The newly created child node.
        """
        new_state = node.state.copy()
        new_state.make_move(action[0], action[1])
        new_state.change_player()
        
        child_node = MCTS_NODE()
        child_node.state = new_state
        child_node.parent = node
        child_node.player = new_state.current_player
        
        # Only add untried actions if this isn't a terminal state
        if new_state.is_game_over() is None:
            child_node.untried_actions = new_state.valid_moves()
        else:
            child_node.untried_actions = []
            
        node.children[action] = child_node
        return child_node

    def _simulate(self, node):
        """
        Simulate a random game from the given node.
        :param node: The node to simulate from.
        :return: The reward from the simulation.
        """
        # If the node is already a terminal state, return its result directly
        result = node.state.is_game_over()
        if result is not None:
            return result
        
        current_state = node.state.copy()
        
        while True:
            result = current_state.is_game_over()
            if result is not None:
                return result
            
            valid_moves = current_state.valid_moves()
            index = np.arange(len(valid_moves))  # Fixed: use np.arange instead of generator expression
            rand_choice = np.random.choice(index)
            action = valid_moves[rand_choice]
            current_state.make_move(action[0], action[1])
            current_state.switch_player()

    def _backpropagate(self, node, reward):
        """
        Backpropagate the reward up the tree.
        :param node: The node to backpropagate from.
        :param reward: The reward to backpropagate.
        """
        while node is not None:
            node.visits += 1
            # White wants positive rewards, Black wants negative rewards
            node.value_sum += reward * node.player 
            node = node.parent

    def _uct_value(self, node):
        """
        Calculate the UCT value for a node.
        :param node: The node to calculate the UCT value for.
        :return: The UCT value.
        """
        if node.visits == 0:
            return float('inf')
        
        # UCT formula: exploitation + exploration
        exploitation = node.value_sum / node.visits
        exploration = np.sqrt(2 * np.log(node.parent.visits) / node.visits)
            
        return -exploitation + exploration # Negate exploitation so we minimize score of opposing player (only for two player zero sum)