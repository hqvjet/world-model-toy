import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, frame_size=20, context_size=3, hidden_size=64):
        super(WorldModel, self).__init__()
        self.frame_size = frame_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.transition = nn.Sequential(
            nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        )
    
    def _compute_loss(self, pred_o_t, o_t, rollout_end):
        criterion = nn.MSELoss()
        # Only compute loss on predicted future states (from context_size to rollout_end)
        loss_wm = criterion(pred_o_t, o_t[:, self.context_size:rollout_end])
        
        # Autoencoder loss on all states
        o_t_expanded = o_t.unsqueeze(-1)
        loss_autoencoder = criterion(self.decoder(self.encoder(o_t_expanded)), o_t_expanded)
        return loss_wm + loss_autoencoder

    def forward(
            self, 
            o_t: torch.Tensor,    # [B, F] - full sequence with context
            rollout_size: int = 17  # Number of future states to predict
        ):
        pred_o_ts = []  # Will contain predictions for future states
        z = []          # Latent states
        
        # Encode context states (first 3 states)
        for t in range(self.context_size):
            z_t = self.encoder(o_t[:,t].unsqueeze(-1))
            z.append(z_t)
        
        rollout_end = min(self.context_size + rollout_size, self.frame_size)
        
        # Predict future states using transition model (up to rollout_end)
        for t in range(self.context_size, rollout_end):
            z_t = self.transition(z[t-1])
            z.append(z_t)
            pred_o_ts.append(self.decoder(z_t).squeeze(-1))
        
        # For remaining states (if any), use ground truth encoding
        for t in range(rollout_end, self.frame_size):
            z_t = self.encoder(o_t[:,t].unsqueeze(-1))
            z.append(z_t)
        
        pred_o_t = torch.stack(pred_o_ts, dim=1)
        loss = self._compute_loss(pred_o_t, o_t, rollout_end)

        return {
            'loss': loss,
            'pred_o_ts': pred_o_t
        }