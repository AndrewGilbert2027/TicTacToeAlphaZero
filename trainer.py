from MCTS_NN import MCTS_Deep
from Model import TicTacToeCNN
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import os
import time

class Trainer:
    def __init__(self, game_class, model=None):
        """
        Initialize the trainer with a game class and model.
        
        Args:
            game_class: Class used to create game instances
            model: Optional pre-trained model to use
        """
        self.game_class = game_class
        
        # Initialize model or use provided one
        if model is None:
            self.model = TicTacToeCNN()
        else:
            self.model = model
            
        # Create save directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

    def get_data(self, num_games=10, temperature=1.0, num_simulations=300, bootstrap=False):
        """
        Collect training data by self-play with MCTS.
        
        Args:
            num_games: Number of complete games to play
            temperature: Temperature for action selection
            num_simulations: Number of MCTS simulations per move
            bootstrap: Whether to use estimated values instead of final outcomes
            
        Returns:
            List of (state_tensor, final_outcome, policy_tensor) tuples
        """
        training_data = []
        
        for game_num in range(num_games):
            print(f"Game {game_num+1}/{num_games}")
            # Create a new game instance instead of trying to call reset as a class method
            game = self.game_class()  # Fixed: create a new instance of the game
            game_history = []
            move_count = 0
            
            # Play until game is over
            while game.is_terminal() is None:
                move_count += 1
                print(f"  Move {move_count}", end="\r")
                
                # Store current board state
                current_state = game.copy()
                
                # Run MCTS to get best move
                mcts = MCTS_Deep(current_state, self.model)
                est_value, visit_counts = mcts.search(num_simulations=num_simulations)
                print(f"  Estimated value: {est_value}, Visit counts: {visit_counts}")
                
                # No valid moves or something went wrong
                if not visit_counts:
                    print("Warning: No valid moves found")
                    break
                
                # Convert visit counts to policy
                policy = self._visits_to_policy(visit_counts, current_state.valid_moves())
                
                # Store state and policy for training
                if bootstrap:
                    # Use bootstrap value if enabled
                    game_history.append((current_state.get_feature_plane(), est_value, policy))
                else:
                    game_history.append((current_state.get_feature_plane(), policy))
                
                # Select move based on visit counts (use temperature=0.1 for more exploitation)
                action = self._select_action(visit_counts, temperature=temperature)
                
                # Apply selected move
                try:
                    game = game.step(action)
                    print(f"  Selected action: {action} (Move {move_count})")
                except Exception as e:
                    print(f"Error applying action {action}: {e}")
                    break
            
            # Game finished, get final outcome
            outcome = game.is_terminal()
            print(f"Game {game_num+1} finished with outcome: {outcome} in {move_count} moves")
        

            # Add all states from this game to training data
            if bootstrap:
                # Use model's estimated values (already in tanh scale)
                for state_tensor, est_value, policy in game_history:
                    training_data.append((state_tensor, est_value, policy))
            else:
                # Use final game outcome (already in tanh scale)
                for state_tensor, policy in game_history:
                    training_data.append((state_tensor, outcome, policy))
                
        return training_data

    def _visits_to_policy(self, visits, valid_moves):
        """
        Convert visit counts to policy tensor.
        
        Args:
            visits: Dictionary mapping actions to visit counts
            valid_moves: List of valid moves
            
        Returns:
            3x3 tensor with normalized visit probabilities
        """
        # Create empty policy tensor
        policy = torch.zeros((3, 3), dtype=torch.float32)
        
        # Calculate total visits for normalization
        total_visits = sum(visits.values())
        
        if total_visits > 0:
            # Fill in normalized visit counts
            for action, count in visits.items():
                row, col = action
                policy[row, col] = count / total_visits
        else:
            # If no visits (shouldn't happen), use uniform distribution
            for move in valid_moves:
                row, col = move
                policy[row, col] = 1.0 / len(valid_moves)
                
        return policy

    def _select_action(self, visits, temperature=1.0):
        """
        Select action based on visit counts and temperature.
        
        Args:
            visits: Dictionary mapping actions to visit counts
            temperature: Controls exploration vs exploitation
            
        Returns:
            Selected action
        """
        if not visits:
            raise ValueError("No valid actions provided")
            
        # Use greedy selection for temperature=0
        if temperature == 0:
            return max(visits.items(), key=lambda x: x[1])[0]
        
        # Get actions and visit counts
        actions = list(visits.keys())
        counts = np.array([visits[action] for action in actions])
        
        # Apply temperature scaling
        if temperature != 1.0:
            counts = counts ** (1.0 / temperature)
            
        # Convert to probabilities
        probs = counts / np.sum(counts)
        
        # Sample action according to probabilities
        try:
            return actions[np.random.choice(len(actions), p=probs)]
        except:
            # Fallback to greedy if something goes wrong
            return actions[np.argmax(counts)]

    def train(self, num_epochs=10, batch_size=32, lr=0.001, num_games=10, bootstrap=False):
        """
        Train the model using self-play data.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            num_games: Number of games to play per epoch
            bootstrap: Whether to use estimated values instead of final outcomes
        """
        temperature = 1.0
        for epoch in range(num_epochs):
            start_time = time.time()
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Collect training data through self-play
            training_data = self.get_data(num_games=num_games, temperature=temperature, bootstrap=bootstrap)
            temperature = max(temperature * 0.9, 0.1)  # Decay temperature

            if not training_data:
                print("No training data collected, skipping epoch")
                continue
                
            # Process training data
            state_tensors = []
            value_targets = []
            policy_targets = []
            
            for state_tensor, outcome, policy_tensor in training_data:
                state_tensors.append(state_tensor)
                # Outcome is already in tanh range (-1 to 1)
                value_targets.append(outcome)
                policy_targets.append(policy_tensor)
            
            # Create tensors
            state_tensors = torch.stack(state_tensors)
            value_targets = torch.tensor(value_targets, dtype=torch.float32).view(-1, 1)
            policy_targets = torch.stack(policy_targets)
            
            # Create dataset and dataloader
            dataset = torch.utils.data.TensorDataset(state_tensors, value_targets, policy_targets)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
            # Set up optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            
            # Training loop
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for states, values, policies in dataloader:
                num_batches += 1
                optimizer.zero_grad()
                
                # Forward pass - we can't use the model's forward method directly with batches
                # So we use the internal network forward method
                # Prepare input tensors
                if states.dim() == 3:  # (batch, height, width)
                    states = states.unsqueeze(1)  # Add channel dimension
                
                # Forward pass through network layers
                pred_values, pred_policies = self.model._network_forward(states)
                
                # Value loss - MSE between predicted and target values
                # Both are already in tanh range (-1 to 1)
                value_loss = F.mse_loss(pred_values, values)
                
                # Policy loss - KL divergence between predicted and target policies
                # Flatten tensors for loss calculation
                pred_policies_flat = F.softmax(pred_policies.view(-1, 9), dim=1)
                target_policies_flat = policies.view(-1, 9)
                
                # Use KL divergence as policy loss
                policy_loss = F.kl_div(
                    pred_policies_flat.log(),  # Input needs to be log probabilities
                    target_policies_flat,
                    reduction='batchmean'
                )
                
                # Combined loss
                loss = value_loss + policy_loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Save model after each epoch
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f} Time: {epoch_time:.1f}s")
            
            # Save model
            self.model.save(f"models/b_model_epoch_{epoch+1}.pth")
    
    def save_model(self, path):
        """Save the model to a file."""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a model from a file."""
        self.model.load(path)
        return self.model


if __name__ == "__main__":
    # Example usage
    from game_engine.ttt import TicTacToe
    
    # Create trainer
    trainer = Trainer(TicTacToe)
    
    # Train model
    trainer.train(num_epochs=30, batch_size=32, num_games=10, bootstrap=True)
    
    # Save final models
    trainer.save_model("models/final_model.pth")