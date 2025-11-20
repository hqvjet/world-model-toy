import torch
import numpy as np
import cv2
from tqdm import tqdm

from models.vae import VAE
from utils import rescale_output

dataset_path = 'datasets'
model_path = 'trained_models'
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VAE model
    vision_model = VAE(latent_dim=32)
    vision_model.load_state_dict(torch.load(f'{model_path}/vision_model.pth', map_location=device))
    vision_model.to(device)
    vision_model.eval()

    # Load all frames
    frames = np.load(f'{dataset_path}/car_racing_memory_dataset_frames.npy')
    print(f"Loaded frames shape: {frames.shape}")
    
    # Flatten all sequences into one continuous stream
    num_sequences, seq_len = frames.shape[0], frames.shape[1]
    total_frames = num_sequences * seq_len
    all_frames = frames.reshape(-1, *frames.shape[2:])  # (total_frames, C, H, W)
    
    print(f"Total frames to process: {total_frames}")
    
    reconstructed_frames = []
    
    print("Reconstructing frames with VAE...")
    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 100
        for i in tqdm(range(0, total_frames, batch_size), desc="VAE Reconstruction"):
            batch = all_frames[i:i+batch_size]
            
            # Normalize and convert to tensor
            batch_tensor = torch.from_numpy(batch).float().to(device) / 127.5 - 1.0
            
            # Encode and decode
            z = vision_model.encode(batch_tensor)['mu']
            reconstructed = vision_model.decode(z)
            
            # Convert back to uint8 images
            reconstructed = rescale_output(reconstructed)  # (batch, H, W, C)
            reconstructed_frames.append(reconstructed.cpu().numpy())
    
    reconstructed_frames = np.concatenate(reconstructed_frames, axis=0)  # (total_frames, H, W, C)
    print(f"Reconstructed frames shape: {reconstructed_frames.shape}")
    
    # Prepare real frames for comparison (convert to H, W, C format)
    real_frames = all_frames.transpose(0, 2, 3, 1)  # (total_frames, H, W, C)
    
    # Create side-by-side comparison video
    output_path = 'figures/vision_comparison_video.mp4'
    frame_height, frame_width = real_frames.shape[1], real_frames.shape[2]
    combined_width = frame_width * 2 + 10  # Add small gap between frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (combined_width, frame_height))
    
    print("Creating comparison video...")
    for i in tqdm(range(total_frames), desc="Writing video"):
        # Create combined frame with gap
        combined = np.ones((frame_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Add real frame on left
        combined[:, :frame_width] = real_frames[i]
        
        # Add reconstructed frame on right
        combined[:, frame_width+10:] = reconstructed_frames[i]
        
        # Add text labels
        cv2.putText(combined, 'Real', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, 'VAE Reconstruction', (frame_width + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)
    
    out.release()
    print(f"Saved comparison video to {output_path}")
    
    # Calculate reconstruction error statistics
    mse = np.mean((real_frames.astype(float) - reconstructed_frames.astype(float)) ** 2)
    print(f"\nReconstruction MSE: {mse:.4f}")
    print(f"Average pixel difference: {np.sqrt(mse):.4f}")
