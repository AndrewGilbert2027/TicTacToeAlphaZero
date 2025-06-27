import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeCNN(nn.Module):
    def __init__(self, num_channels=32):
        """Initialize the Tic-Tac-Toe CNN model."""
        super(TicTacToeCNN, self).__init__()
        
        # Input: 1 x 3 x 3 (1 channel for the board state)
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # Policy head - outputs probabilities for 9 positions
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 3 * 3, 9)  # Flattened to 9 positions
        
        # Value head - outputs scalar evaluation
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(3 * 3, 32)
        self.value_fc2 = nn.Linear(32, 1)

        self.policy_shape = (3, 3)  # Shape of the policy output
        self.value_shape = (1,)     # Shape of the value output

    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Game state object with get_feature_plane method
                  or tensor
        
        Returns:
            value: Scalar value estimate (-1 to 1)
            policy: Dictionary mapping (row, col) positions to probabilities
        """
        # Process input
        if isinstance(state, torch.Tensor):
            x = self._prepare_tensor(state)
            # Use default valid moves
            valid_moves = [(i, j) for i in range(3) for j in range(3)]
        else:
            # Get feature plane from game state
            try:
                feature_plane = state.get_feature_plane()  # Should be shape (1, 3, 3)
                x = self._prepare_tensor(feature_plane)
                valid_moves = state.valid_moves()
            except AttributeError:
                raise ValueError("Input state must have get_feature_plane and valid_moves methods")

        # Run neural network
        value, policy_logits = self._network_forward(x)
        
        # Convert policy logits to dictionary of probabilities
        policy_dict = self._create_policy_dict(policy_logits, valid_moves)
        
        # Return scalar value and policy dictionary
        return value.item(), policy_dict

    def _prepare_tensor(self, tensor):
        """
        Prepare input tensor for the network.
        
        Args:
            tensor: Input tensor of any valid board shape
            
        Returns:
            Tensor of shape (batch_size, channels, height, width)
        """
        # Handle different input shapes
        if tensor.dim() == 4:  # Already (batch, channels, height, width)
            return tensor
            
        if tensor.dim() == 3:
            # Assume it's (channels, height, width)
            return tensor.unsqueeze(0)  # Add batch dimension
            
        if tensor.dim() == 2:  # (height, width)
            return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}. Must be 2D, 3D or 4D.")

    def _network_forward(self, x):
        """
        Internal network forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            value: Value output tensor in range [-1, 1] (using tanh activation)
            policy: Policy output tensor
        """
        # Shared representation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(x.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)  # Linear layer to get logits
        policy = policy.view(x.size(0), 3, 3)  # Reshape to board dimensions
        
        # Value head - outputs value in range [-1, 1]
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Tanh activation for -1 to 1 range
        
        return value, policy

    def _create_policy_dict(self, policy_tensor, valid_moves):
        """Convert policy tensor to dictionary of probabilities."""
        # Apply softmax to get probabilities
        policy_probs = F.softmax(policy_tensor.view(policy_tensor.size(0), -1), dim=1)
        policy_probs = policy_probs.view(policy_tensor.size(0), 3, 3)
        
        # Create mask for valid moves
        mask = torch.zeros_like(policy_probs)
        for move in valid_moves:
            row, col = move
            mask[0, row, col] = 1.0
        
        # Apply mask and renormalize
        masked_policy = policy_probs * mask
        total_prob = masked_policy.sum()
        
        # Handle case where no valid moves have probability
        if total_prob < 1e-10:
            # Uniform distribution over valid moves
            for move in valid_moves:
                row, col = move
                masked_policy[0, row, col] = 1.0 / len(valid_moves)
        else:
            # Normalize to sum to 1
            masked_policy = masked_policy / total_prob
        
        # Create dictionary
        policy_dict = {}
        for move in valid_moves:
            row, col = move
            policy_dict[move] = masked_policy[0, row, col].item()
            
        return policy_dict
    
    def get_value(self, state):
        """Get value evaluation for a state."""
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                x = self._prepare_tensor(state)
            else:
                try:
                    feature_plane = state.get_feature_plane()
                    x = self._prepare_tensor(feature_plane)
                except AttributeError:
                    raise ValueError("Input state must have get_feature_plane method")
            
            # Forward pass through network
            value, _ = self._network_forward(x)
            
            # Return scalar value
            return value.item()

    def save(self, path):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))
        self.eval()

# Utility functions
def preprocess_board(board):
    """Convert numpy board to tensor suitable for the model."""
    return torch.FloatTensor(board).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

def get_move_from_policy(policy_dict):
    """Select the move with highest probability from policy dictionary."""
    return max(policy_dict.items(), key=lambda x: x[1])[0]