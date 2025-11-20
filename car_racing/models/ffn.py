import torch
import torch.nn as nn

class ControlModel(nn.Module):
    """
    Simple linear controller as in Ha & Schmidhuber 2018 (World Models)
    Maps (z, h) → action with a single linear layer + tanh activation
    """
    def __init__(self, latent_dim, hidden_dim, output_dim=3):
        super(ControlModel, self).__init__()
        # Single linear layer (no hidden layers) as per World Models paper
        self.fc = nn.Linear(latent_dim + hidden_dim, output_dim)
        
    def forward(self, z, h):
        """
        Args:
            z: latent state (batch, latent_dim)
            h: hidden state (batch, hidden_dim)
        Returns:
            action: (batch, 3) with steering in [-1,1], gas/brake in [0,1]
        """
        x = torch.cat([z, h], dim=-1)  # Concatenate latent and hidden state
        x = self.fc(x)
        
        # Apply activations as per CarRacing action space
        steering = torch.tanh(x[:, 0])  # [-1, 1]
        gas = torch.sigmoid(x[:, 1])  # [0, 1]
        brake = torch.sigmoid(x[:, 2])  # [0, 1]
        
        action = torch.stack([steering, gas, brake], dim=-1)
        
        return action