import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from model import WorldModel
from utils import generate_samples, visualize_movement


epochs = 100
if __name__ == "__main__":
    o_t = generate_samples(n=5000, frame_size=20)
    
    # Apply standard scaling (normalization)
    mean = o_t.mean()
    std = o_t.std()
    o_t = (o_t - mean) / std

    # visualize_movement(o_t[0])
    
    dataloader = DataLoader(o_t, batch_size=128, shuffle=True)
    model = WorldModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    best_loss = float('inf')
    for epoch in range(epochs):
        rollout_size = min(17, epoch//5 + 1)  # Gradually increase from 1 to 17
        for sample in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            output = model(sample, rollout_size=rollout_size) 
            loss = output['loss']
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
        
        if loss.item() < best_loss and rollout_size == 17:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved new best model with loss: {best_loss:.4f}")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f} | Rollout Size: {rollout_size}")