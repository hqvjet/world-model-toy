import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    VAE model based on World Models (Ha & Schmidhuber, 2018)
    Encodes 96x96 RGB images to latent dimension
    Architecture: 4 Conv layers -> FC -> Latent -> FC -> 4 Deconv layers
    """
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 96x96x3 -> 6x6x256
        self.encoder = nn.Sequential(
            # 96x96x3 -> 48x48x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 48x48x32 -> 24x24x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 24x24x64 -> 12x12x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 12x12x128 -> 6x6x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Flatten size: 256 * 6 * 6 = 9216
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 1024)
        
        self.decoder = nn.Sequential(
            # Reshape to 1024x1x1, then upsample
            # 1x1x1024 -> 6x6x128
            nn.ConvTranspose2d(1024, 128, kernel_size=6, stride=1),
            nn.ReLU(),
            # 6x6x128 -> 12x12x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 12x12x64 -> 24x24x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 24x24x32 -> 48x48x16
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 48x48x16 -> 96x96x3
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Normalize input from [-1, 1] to [0, 1] for encoder
        x_norm = (x + 1.0) / 2.0
        
        # Encode
        h = self.encoder(x_norm)
        h = h.reshape(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent numerical instability
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        h = self.fc_decode(z)
        h = h.reshape(h.size(0), 1024, 1, 1)  # Reshape for conv layers
        reconstruction = self.decoder(h)
        
        # Convert output from [0, 1] back to [-1, 1]
        reconstruction = reconstruction * 2.0 - 1.0
        
        return {
            'reconstruction': reconstruction,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def encode(self, x):
        """Encode input to latent space"""
        # Normalize input from [-1, 1] to [0, 1]
        x_norm = (x + 1.0) / 2.0
        
        h = self.encoder(x_norm)
        h = h.reshape(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return {'mu': mu, 'logvar': logvar}
    
    def decode(self, z):
        """Decode latent vector to image"""
        h = self.fc_decode(z)
        h = h.reshape(h.size(0), 1024, 1, 1)
        reconstruction = self.decoder(h)
        # Convert from [0, 1] to [-1, 1]
        reconstruction = reconstruction * 2.0 - 1.0
        return reconstruction