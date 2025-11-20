import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN (Ha & Schmidhuber, 2018)
    Predicts future latent states given current latent + action
    Uses LSTM (not GRU) as per World Models paper
    """
    def __init__(self, latent_dim=32, action_dim=3, hidden_size=256, num_layers=1, num_mixtures=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        
        # LSTM instead of GRU (as per World Models paper)
        self.lstm = nn.LSTM(
            latent_dim + action_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True
        )
        
        # MDN output heads
        self.fc_pi = nn.Linear(hidden_size, num_mixtures)
        self.fc_mu = nn.Linear(hidden_size, num_mixtures * latent_dim)
        self.fc_sigma = nn.Linear(hidden_size, num_mixtures * latent_dim)
        
    def forward(self, z, a, hidden=None):
        """
        Args:
            z: latent state (batch, seq_len, latent_dim)
            a: action (batch, seq_len, action_dim)
            hidden: (h, c) tuple for LSTM
        Returns:
            dict with 'pi', 'mu', 'sigma', 'hidden'
        """
        # Concatenate latent and action
        x = torch.cat([z, a], dim=-1)  # (batch, seq_len, latent_dim + action_dim)
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch, seq_len, hidden_size)
        
        # MDN parameters
        pi = self.fc_pi(lstm_out)  # (batch, seq_len, num_mixtures)
        mu = self.fc_mu(lstm_out)  # (batch, seq_len, num_mixtures * latent_dim)
        sigma = self.fc_sigma(lstm_out)  # (batch, seq_len, num_mixtures * latent_dim)
        
        # Reshape mu and sigma
        batch_size, seq_len = mu.size(0), mu.size(1)
        mu = mu.reshape(batch_size, seq_len, self.num_mixtures, self.latent_dim)
        sigma = sigma.reshape(batch_size, seq_len, self.num_mixtures, self.latent_dim)
        
        # Apply activations with clamping to prevent numerical issues
        sigma = torch.exp(sigma.clamp(min=-10, max=10))  # Clamp before exp to avoid underflow/overflow
        sigma = sigma.clamp(min=1e-4)  # Ensure sigma >= 0.0001 to prevent log(0)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'hidden': hidden
        }
    
    def compute_mdn_loss(self, target, pi, mu, sigma):
        """
        Compute MDN loss (negative log-likelihood)
        Args:
            target: ground truth next latent (batch, seq_len, latent_dim)
            pi: mixture weights (batch, seq_len, num_mixtures)
            mu: mixture means (batch, seq_len, num_mixtures, latent_dim)
            sigma: mixture std devs (batch, seq_len, num_mixtures, latent_dim)
        """
        batch_size, seq_len, latent_dim = target.size()
        
        # Expand target for all mixtures
        target = target.unsqueeze(2)  # (batch, seq_len, 1, latent_dim)
        target = target.expand(-1, -1, self.num_mixtures, -1)  # (batch, seq_len, num_mixtures, latent_dim)
        
        # Compute Gaussian log probabilities for each mixture
        # log N(x|mu,sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((x-mu)/sigma)^2
        log_2pi = math.log(2.0 * math.pi)
        
        # Per-dimension log prob
        log_prob_per_dim = -0.5 * log_2pi - torch.log(sigma) - 0.5 * ((target - mu) / sigma) ** 2
        
        # Sum over latent dimensions
        log_prob = torch.sum(log_prob_per_dim, dim=-1)  # (batch, seq_len, num_mixtures)
        
        # Add mixture weights (log-sum-exp trick for numerical stability)
        log_pi = F.log_softmax(pi, dim=-1)  # (batch, seq_len, num_mixtures)
        log_mix_prob = log_pi + log_prob  # (batch, seq_len, num_mixtures)
        
        # Log-sum-exp over mixtures
        max_log_mix = torch.max(log_mix_prob, dim=-1, keepdim=True)[0]
        log_sum_exp = max_log_mix + torch.log(torch.sum(torch.exp(log_mix_prob - max_log_mix), dim=-1, keepdim=True))
        log_sum_exp = log_sum_exp.squeeze(-1)  # (batch, seq_len)
        
        # Negative log likelihood (average over batch and sequence)
        nll = -torch.mean(log_sum_exp)
        
        return nll
    
    def sample(self, pi, mu, sigma, temperature=1.0):
        """
        Sample from the MDN
        Args:
            pi: mixture weights (batch, seq_len, num_mixtures)
            mu: mixture means (batch, seq_len, num_mixtures, latent_dim)
            sigma: mixture std devs (batch, seq_len, num_mixtures, latent_dim)
            temperature: sampling temperature (higher = more random)
        Returns:
            sampled latent (batch, seq_len, latent_dim)
        """
        batch_size, seq_len = pi.size(0), pi.size(1)
        
        # Apply temperature to mixture weights
        pi = F.softmax(pi / temperature, dim=-1)  # (batch, seq_len, num_mixtures)
        
        # Flatten for sampling
        pi_flat = pi.reshape(-1, self.num_mixtures)  # (batch*seq_len, num_mixtures)
        mu_flat = mu.reshape(-1, self.num_mixtures, self.latent_dim)  # (batch*seq_len, num_mixtures, latent_dim)
        sigma_flat = sigma.reshape(-1, self.num_mixtures, self.latent_dim)  # (batch*seq_len, num_mixtures, latent_dim)
        
        # Sample mixture component for each timestep
        mixture_idx = torch.multinomial(pi_flat, 1).squeeze(-1)  # (batch*seq_len,)
        
        # Gather selected mixture parameters
        batch_indices = torch.arange(pi_flat.size(0), device=pi.device)
        selected_mu = mu_flat[batch_indices, mixture_idx]  # (batch*seq_len, latent_dim)
        selected_sigma = sigma_flat[batch_indices, mixture_idx]  # (batch*seq_len, latent_dim)
        
        # Sample from selected Gaussian
        eps = torch.randn_like(selected_mu)
        sample = selected_mu + selected_sigma * eps
        
        # Reshape back
        sample = sample.reshape(batch_size, seq_len, self.latent_dim)
        
        return sample
