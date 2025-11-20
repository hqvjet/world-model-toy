import torch
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from models.mdn_rnn import MDNRNN
from models.vae import VAE

def prepare_frames_batch(frames_batch):
    """Prepare frames batch for vision model encoding"""
    # frames_batch shape: (batch_size, seq_len, channels, height, width) - already preprocessed
    # Just need to flatten the sequence dimension
    batch_size, seq_len = frames_batch.shape[0], frames_batch.shape[1]
    # Reshape to (batch_size * seq_len, channels, height, width)
    frames_flat = frames_batch.reshape(-1, *frames_batch.shape[2:])
    # Already in correct format (C, H, W), just need to convert to float and normalize
    frames_tensor = frames_flat.float() / 127.5 - 1.0  # Normalize to [-1, 1]
    return frames_tensor, batch_size, seq_len

dataset_path = 'datasets'
model_path = 'trained_models'
if __name__ == "__main__":
    actions = np.load(dataset_path + '/car_racing_memory_dataset_actions.npy')
    frames = np.load(dataset_path + '/car_racing_memory_dataset_frames.npy')
    print(f"Loaded actions shape: {actions.shape}")
    print(f"Loaded frames shape: {frames.shape}")
    print(f"Frame dtype: {frames.dtype}")
    print(f"Single frame shape: {frames[0].shape if len(frames.shape) > 1 else 'N/A'}")

    # Improved hyperparameters
    dataloader = DataLoader(list(zip(frames, actions)), batch_size=32, shuffle=True)
    memory_model = MDNRNN(latent_dim=32, action_dim=3, hidden_size=512, num_layers=1, num_mixtures=5)  # Larger hidden size
    vision_model = VAE(latent_dim=32)
    vision_model.load_state_dict(torch.load(f'{model_path}/vision_model.pth'))
    vision_model.eval()
    
    # Better optimizer settings
    optimizer = torch.optim.Adam(params=memory_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    accelerator = Accelerator()
    memory_model, vision_model, optimizer, dataloader, scheduler = accelerator.prepare(
        memory_model, vision_model, optimizer, dataloader, scheduler
    )

    num_epochs = 1000  # More epochs for temporal learning
    patient = 50  # More patience
    patience_counter = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        memory_model.train()
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        
        for batch_idx, (batch_frames, batch_actions) in enumerate(loop):
            optimizer.zero_grad()
            
            with torch.no_grad():
                frames_scaled, batch_size, seq_len = prepare_frames_batch(batch_frames)
                encoded = vision_model.encode(frames_scaled)
                z = encoded['mu']
                # Reshape back to (batch_size, seq_len, latent_dim)
                z = z.reshape(batch_size, seq_len, -1)
                
                # Print latent stats for first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"\nLatent stats - Mean: {z.mean():.4f}, Std: {z.std():.4f}, Min: {z.min():.4f}, Max: {z.max():.4f}")
            
            a = batch_actions.float()
            
            # Forward pass through memory model
            outputs = memory_model(z=z[:, :-1, :], a=a[:, :-1, :], hidden=None)
            
            # Compute loss
            loss = memory_model.compute_mdn_loss(
                target=z[:, 1:, :],  # Target: next latent states
                pi=outputs['pi'],
                mu=outputs['mu'],
                sigma=outputs['sigma']
            )
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Gradient clipping for stability (looser for better learning)
            accelerator.backward(loss)
            grad_norm = torch.nn.utils.clip_grad_norm_(memory_model.parameters(), max_norm=5.0)
            
            # Monitor gradients more frequently
            if batch_idx == 0:
                print(f"Epoch {epoch+1} - Gradient norm: {grad_norm:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=epoch_loss/(loop.n+1))

        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {epoch_loss:.6f} - Best: {best_loss:.6f}")
        
        if epoch_loss < best_loss:
            patience_counter = 0
            best_loss = epoch_loss
            unwrapped_model = accelerator.unwrap_model(memory_model)
            torch.save(unwrapped_model.state_dict(), model_path + '/memory_model.pth')
            print(f"✓ Saved new best model with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patient:
                print("Early stopping triggered.")
                break