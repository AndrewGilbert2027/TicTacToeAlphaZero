# Tic-Tac-Toe Reinforcement Learning Project

This project implements a Tic-Tac-Toe game with two types of AI opponents:
1. A classic Monte Carlo Tree Search (MCTS) algorithm
2. A neural network enhanced MCTS algorithm (AlphaZero-style)

The neural network-based agent is trained through self-play reinforcement learning to master the game of Tic-Tac-Toe.

## Repository Structure

- `game_engine/ttt.py` - Tic-Tac-Toe game implementation
- `MCTS.py` - Classic Monte Carlo Tree Search implementation
- `MCTS_NN.py` - Neural network enhanced MCTS (similar to AlphaZero)
- `Model.py` - Neural network model (CNN-based policy and value networks)
- `trainer.py` - Self-play training logic for the neural network
- `game_manager_gui.py` - GUI for playing against the AI
- `tests/` - Test files for the various components

## Setup

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pygame (for the GUI)

Install the required packages:

```bash
pip install torch numpy pygame pytest
```

## Running the Game

To play against the AI with a graphical interface:

### Play against the regular MCTS algorithm
```bash
python game_manager_gui.py
```

### Play against the neural network-enhanced MCTS
```bash
python game_manager_gui.py --deep
```

## Training the Model
To train the neural network through self-play:
```bash
# Basic training with default parameters
python trainer.py
```

### Advanced training with customized parameters
```bash
# Play against the neural network enhanced architecture
python trainer.py --epochs 50 --games 100 --simulations 200
```

The trained models will be saved in the models/ directory.

## Running Tests
To run all tests:
```bash
pytest
```

## Algorithms
#### Classic MCTS
The classic Monte Carlo Tree Search uses four phases:

Selection: Traverse the tree using PUCT formula
Expansion: Add a new node to the tree
Simulation: Play a random game from the new node
Backpropagation: Update node statistics based on the game outcome

#### Neural Network Enhanced MCTS
Similar to AlphaZero, this version:

1. Uses a neural network to evaluate board positions instead of random simulations
2. The network outputs both a value estimate and a policy (probability distribution over moves)
3. Uses PUCT formula for tree traversal, incorporating the policy from the neural network
4. Trained through self-play and iterative improvement (option to choose bootstrap method)






