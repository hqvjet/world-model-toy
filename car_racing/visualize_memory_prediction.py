import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from models.mdn_rnn import MDNRNN
from models.vae import VAE
from utils import rescale_output

dataset_path = 'datasets'
model_path = 'trained_models'
figures_path = 'figures'
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    memory_model = MDNRNN(latent_dim=32, action_dim=3, hidden_size=512, num_layers=1, num_mixtures=5)  # Match training config
    vision_model = VAE(latent_dim=32)
    vision_model.load_state_dict(torch.load(f'{model_path}/vision_model.pth', map_location=device))
    vision_model.to(device)
    vision_model.eval()
    memory_model.load_state_dict(torch.load(f'{model_path}/memory_model.pth', map_location=device))
    memory_model.to(device)
    memory_model.eval()

    # Load initial data
    frames = np.load(f'{dataset_path}/car_racing_memory_dataset_frames.npy')
    actions = np.load(f'{dataset_path}/car_racing_memory_dataset_actions.npy')
    print(f"Loaded frames shape: {frames.shape}")
    print(f"Loaded actions shape: {actions.shape}")

    # Predict all frames from the entire dataset
    num_sequences, seq_len = frames.shape[0], frames.shape[1]
    SEED_LEN = 30
    ROLLOUT_LEN = 50  # Dream for 50 steps, then sync with real world
    
    # We'll use first SEED_LEN frames as warmup, then predict the rest
    total_frames = num_sequences * seq_len
    GEN_LEN = total_frames - SEED_LEN
    
    # Flatten all frames and actions
    all_frames = frames.reshape(-1, *frames.shape[2:])  # (total_frames, C, H, W)
    all_actions = actions.reshape(-1, actions.shape[-1])  # (total_frames, 3)
    
    seed_frames = all_frames[:SEED_LEN]  # (SEED_LEN, C, H, W)
    seed_actions = all_actions[:SEED_LEN]  # (SEED_LEN, 3)
    
    print(f"Total frames: {total_frames}, Seed: {SEED_LEN}, Generating: {GEN_LEN}")
    print(f"Rollout strategy: Dream {ROLLOUT_LEN} steps → Sync with real world → Repeat")
    
    dream_frames = []
    hidden = None
    
    print("Generating dream sequence with periodic reality checks...")
    with torch.no_grad():
        # Encode seed frames to get initial latent state
        seed_frames_tensor = torch.from_numpy(seed_frames).float().to(device) / 127.5 - 1.0
        seed_actions_tensor = torch.from_numpy(seed_actions).float().to(device)
        
        # Get initial latent representations
        z_seed = vision_model.encode(seed_frames_tensor)['mu']  # (SEED_LEN, latent_dim)
        z_seed = z_seed.unsqueeze(0)  # (1, SEED_LEN, latent_dim)
        seed_actions_tensor = seed_actions_tensor.unsqueeze(0)  # (1, SEED_LEN, 3)
        
        # Warm up LSTM with seed sequence
        outputs = memory_model(z=z_seed, a=seed_actions_tensor, hidden=hidden)
        hidden = outputs['hidden']
        
        # Start dreaming from last seed frame
        current_z = z_seed[:, -1:, :]  # (1, 1, latent_dim)
        
        current_idx = SEED_LEN
        rollout_counter = 0
        
        pbar = tqdm(total=GEN_LEN, desc="Dreaming with reality sync")
        
        while current_idx < total_frames:
            # Get action for this timestep
            current_action = torch.from_numpy(all_actions[current_idx]).float().to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            
            # Predict next latent state
            outputs = memory_model(z=current_z, a=current_action, hidden=hidden)
            hidden = outputs['hidden']
            
            # Sample from MDN to get next latent state
            next_z = memory_model.sample(
                pi=outputs['pi'][:, -1:, :],  # (1, 1, num_mixtures)
                mu=outputs['mu'][:, -1:, :, :],  # (1, 1, num_mixtures, latent_dim)
                sigma=outputs['sigma'][:, -1:, :, :],  # (1, 1, num_mixtures, latent_dim)
                temperature=1.0  # Higher temperature for more diverse, realistic predictions
            )  # (1, 1, latent_dim)
            
            # Decode to image - squeeze to (1, latent_dim) for decode
            dream_frame = vision_model.decode(next_z.squeeze(1))  # Input: (1, latent_dim) -> Output: (1, C, H, W)
            dream_frame = rescale_output(dream_frame)  # Convert to uint8 image
            dream_frames.append(dream_frame.cpu().numpy()[0])  # (H, W, C)
            
            current_idx += 1
            rollout_counter += 1
            pbar.update(1)
            
            # Every ROLLOUT_LEN steps, sync with real world
            if rollout_counter >= ROLLOUT_LEN and current_idx < total_frames:
                # Get real frame from dataset and encode it
                real_frame = all_frames[current_idx]  # (C, H, W)
                real_frame_tensor = torch.from_numpy(real_frame).float().to(device).unsqueeze(0) / 127.5 - 1.0  # (1, C, H, W)
                real_z = vision_model.encode(real_frame_tensor)['mu']  # (1, latent_dim)
                
                # Reset current_z to real world latent
                current_z = real_z.unsqueeze(1)  # (1, 1, latent_dim)
                
                # Also feed this real latent through LSTM to update hidden state
                real_action = torch.from_numpy(all_actions[current_idx]).float().to(device).unsqueeze(0).unsqueeze(0)
                outputs = memory_model(z=current_z, a=real_action, hidden=hidden)
                hidden = outputs['hidden']
                
                rollout_counter = 0  # Reset counter
                pbar.set_description(f"Synced at frame {current_idx}")
            else:
                # Use predicted latent as input for next step
                current_z = next_z
        
        pbar.close()

    dream_frames = np.array(dream_frames)  # (GEN_LEN, H, W, C)
    print(f"Generated dream frames shape: {dream_frames.shape}")
    
    # Get real frames for comparison (all frames after seed)
    real_frames = all_frames[SEED_LEN:].transpose(0, 2, 3, 1)  # (GEN_LEN, H, W, C)
    print(f"Real frames shape: {real_frames.shape}")
    
    # Both should have the same length now
    comparison_len = len(dream_frames)
    dream_frames_compare = dream_frames
    print(f"Comparison video will be {comparison_len} frames")
    
    # Create side-by-side video
    output_path = f'{figures_path}/memory_comparison_video.mp4'
    frame_height, frame_width = dream_frames_compare.shape[1], dream_frames_compare.shape[2]
    combined_width = frame_width * 2 + 10  # Add small gap between frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (combined_width, frame_height))
    
    print("Creating comparison video...")
    for i in tqdm(range(comparison_len), desc="Writing video"):
        # Create combined frame with gap
        combined = np.ones((frame_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Add real frame on left
        combined[:, :frame_width] = real_frames[i]
        
        # Add dream frame on right
        combined[:, frame_width+10:] = dream_frames_compare[i]
        
        # Add text labels
        cv2.putText(combined, 'Real', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, 'Dream', (frame_width + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
    
    out.release()
    print(f"Saved comparison video to {output_path}")
    
    # Also save some sample comparison frames
    num_samples = 5
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Real vs Dream Frames Comparison', fontsize=16)
    
    for i in range(num_samples):
        frame_idx = i * (comparison_len // num_samples)
        
        # Real frames on top row
        axes[0, i].imshow(real_frames[frame_idx])
        axes[0, i].set_title(f"Real - Frame {frame_idx}")
        axes[0, i].axis('off')
        
        # Dream frames on bottom row
        axes[1, i].imshow(dream_frames_compare[frame_idx])
        axes[1, i].set_title(f"Dream - Frame {frame_idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{figures_path}/memory_comparison_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved sample comparison frames to {figures_path}/memory_comparison_samples.png")