import torch
import gymnasium as gym
from accelerate import Accelerator

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.ffn import ControlModel


env = gym.make(
    "CarRacing-v3", 
    render_mode='human',  # Changed to 'human' to show the game window
    lap_complete_percent=0.95, 
    domain_randomize=False,
    continuous=True
)

model_path = 'trained_models'
if __name__ == "__main__":
    vision_model = VAE(latent_dim=32)
    memory_model = MDNRNN(latent_dim=32, action_dim=3, hidden_size=512, num_layers=1, num_mixtures=5)
    control_model = ControlModel(
        latent_dim=32, 
        hidden_dim=512, 
        output_dim=3
    )
    
    vision_model.load_state_dict(torch.load(f'{model_path}/vision_model.pth'))
    memory_model.load_state_dict(torch.load(f'{model_path}/memory_model.pth'))
    control_model.load_state_dict(torch.load(f'{model_path}/control_model_cmaes.pth'))
    vision_model.eval()
    memory_model.eval()
    control_model.eval()

    accelerator = Accelerator()
    vision_model, memory_model, control_model = accelerator.prepare(
        vision_model, memory_model, control_model
    )

    episodes = 10

    for epi in range(episodes):
        obs, info = env.reset()
        
        # Initialize hidden state at start of episode
        hidden = None
        prev_action = torch.zeros((1, 1, 3)).to(accelerator.device)  # (batch=1, seq=1, action_dim=3)
        
        # Action smoothing to reduce jitter (exponential moving average)
        smoothed_action = torch.zeros(3).to(accelerator.device)
        alpha = 0.3  # Smoothing factor (0 = full smoothing, 1 = no smoothing)
        
        total_reward = 0
        step = 0
        done = False
        
        print(f"\n=== Episode {epi+1}/{episodes} ===")
        
        while not done:
            # 1. Encode current observation with vision model
            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(accelerator.device)  # (1, C, H, W)
            obs_normalized = obs_tensor / 127.5 - 1.0
            
            with torch.no_grad():
                vision_out = vision_model.encode(obs_normalized)
                z = vision_out['mu'].unsqueeze(1)  # (1, 1, latent_dim) - add seq dimension
                
                # 2. Update memory model with current latent + previous action
                memory_out = memory_model(z=z, a=prev_action, hidden=hidden)
                hidden = memory_out['hidden']  # Persist hidden state for next step
                
                # Extract hidden state for control (take last layer)
                h_tensor, c_tensor = hidden
                h_last = h_tensor[-1:, :, :].permute(1, 0, 2)  # (batch=1, 1, hidden_dim)
                
                # 3. Control model predicts action from current latent + hidden state
                z_squeeze = z.squeeze(1)  # (1, latent_dim)
                h_squeeze = h_last.squeeze(1)  # (1, hidden_dim)
                action_pred = control_model(z=z_squeeze, h=h_squeeze)  # (1, 3)
                
                # Apply action smoothing (exponential moving average)
                raw_action = action_pred[0]
                smoothed_action = alpha * raw_action + (1 - alpha) * smoothed_action
                
                # Prepare action for environment (numpy array)
                action = smoothed_action.cpu().numpy()
                
                # Prepare action for next memory update (use smoothed action)
                prev_action = smoothed_action.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            
            # 4. Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            done = terminated or truncated
            
            # Optional: print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}, Reward: {total_reward:.2f}")
        
        print(f"Episode {epi+1} finished: {step} steps, Total Reward: {total_reward:.2f}")
    
    env.close()