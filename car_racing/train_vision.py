import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.vae import VAE
from utils import scale_input

dataset_path = 'datasets'
model_path = 'trained_models'
if __name__ == "__main__":
    np_arr = np.load(dataset_path + '/car_racing_recorded_frames.npy')
    print(f"Loaded frames shape: {np_arr.shape}")

    model = VAE(latent_dim=32)  # Match the new default
    dataloader = DataLoader(np_arr, batch_size=100, shuffle=True)  # Larger batch size for stability
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR, add weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )
    accelerator = Accelerator()
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    num_epochs = 1000
    best_loss = float('inf')
    patient = 30
    patience_counter = 0
    
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        for batch in loop:
            optimizer.zero_grad()
            batch = scale_input(batch)
            outputs = model(batch)
            
            # Reconstruction loss (MSE is better than BCE for continuous outputs)
            recon_loss = torch.nn.functional.mse_loss(outputs['reconstruction'], batch, reduction='sum') / batch.size(0)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()) / batch.size(0)
            
            # Combined loss with KL annealing (World Models uses beta=1.0 after warmup)
            kl_weight = min(1.0, (epoch + 1) / 100.0)  # Gradual warmup over 100 epochs
            loss = recon_loss + kl_weight * kl_loss
            
            accelerator.backward(loss)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            loop.set_postfix(
                loss=loss.item(), 
                recon=recon_loss.item(), 
                kl=kl_loss.item(),
                kl_w=kl_weight
            )

        # Calculate average losses
        epoch_loss /= len(dataloader)
        epoch_recon_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f} | Recon: {epoch_recon_loss:.6f} | KL: {epoch_kl_loss:.6f}")
        
        scheduler.step(epoch_loss)  # Update learning rate based on epoch loss
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), model_path + '/vision_model.pth')
            print(f"✓ Saved best model with loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patient:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break