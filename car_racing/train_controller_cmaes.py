import torch
import numpy as np
from tqdm import tqdm
import cma

from models.ffn import ControlModel
from models.vae import VAE
from models.mdn_rnn import MDNRNN


class WorldModelEnv:
    """Simulate environment using trained V and M models"""
    def __init__(self, vision_model, memory_model, device):
        self.vision_model = vision_model
        self.memory_model = memory_model
        self.device = device
        self.hidden = None
        self.current_z = None
        
    def reset(self, initial_frame):
        """Reset with a real frame from dataset"""
        # Encode initial frame
        frame_tensor = torch.from_numpy(initial_frame).float().to(self.device).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            z = self.vision_model.encode(frame_tensor)['mu']
            self.current_z = z
            self.hidden = None
        return self.current_z
    
    def step(self, action):
        """Predict next state using memory model, assume constant positive reward"""
        with torch.no_grad():
            # Prepare inputs
            z_seq = self.current_z.unsqueeze(1)  # (1, 1, latent_dim)
            a_seq = torch.from_numpy(action).float().to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            
            # Predict next latent
            outputs = self.memory_model(z=z_seq, a=a_seq, hidden=self.hidden)
            self.hidden = outputs['hidden']
            
            # Sample next latent state
            next_z = self.memory_model.sample(
                pi=outputs['pi'],
                mu=outputs['mu'],
                sigma=outputs['sigma'],
                temperature=1.0
            )
            
            self.current_z = next_z.squeeze(1)  # (1, latent_dim)
            
            # Simple reward: penalize drift, encourage stability
            z_mean = self.current_z.mean().item()
            z_std = self.current_z.std().item()
            # Reward for staying near center of latent space (near 0 mean, ~1 std)
            reward = 1.0 - 0.5 * abs(z_mean) - 0.1 * abs(z_std - 1.0)
            
        return self.current_z, reward


def rollout_in_dream(control_model, world_env, initial_frames, num_steps=100, temperature=1.0):
    """
    Rollout controller in the dream (world model)
    Returns: total reward
    """
    total_reward = 0
    
    # Randomly select an initial frame
    idx = np.random.randint(len(initial_frames))
    initial_frame = initial_frames[idx]
    
    # Reset environment
    z = world_env.reset(initial_frame)
    
    # Extract hidden state
    if world_env.hidden is not None:
        h_tensor, c_tensor = world_env.hidden
        h = h_tensor[-1:, :, :].permute(1, 0, 2).squeeze(1)  # (1, hidden_dim)
    else:
        h = torch.zeros(1, 512).to(world_env.device)
    
    for step in range(num_steps):
        # Get action from controller
        with torch.no_grad():
            action_tensor = control_model(z=z, h=h)
            action = action_tensor.cpu().numpy()[0]
        
        # Step in world model
        next_z, reward = world_env.step(action)
        
        # Update hidden state
        if world_env.hidden is not None:
            h_tensor, c_tensor = world_env.hidden
            h = h_tensor[-1:, :, :].permute(1, 0, 2).squeeze(1)
        
        total_reward += reward
        z = next_z
    
    return total_reward


def evaluate_in_real_env(control_model, vision_model, memory_model, device, num_episodes=5):
    """
    Evaluate controller in the real CarRacing environment
    Returns: average reward
    """
    import gymnasium as gym
    
    env = gym.make(
        "CarRacing-v3",
        render_mode='rgb_array',
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True
    )
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        hidden = None
        episode_reward = 0
        prev_action = None
        
        for step in range(500):
            # Encode observation
            obs_tensor = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
            
            with torch.no_grad():
                z = vision_model.encode(obs_tensor)['mu']  # (1, latent_dim)
                
                # Update memory if we have previous action
                if prev_action is not None:
                    z_seq = z.unsqueeze(1)  # (1, 1, latent_dim)
                    a_seq = prev_action.unsqueeze(1)  # (1, 1, 3)
                    mem_out = memory_model(z=z_seq, a=a_seq, hidden=hidden)
                    hidden = mem_out['hidden']
                
                # Get hidden state
                if hidden is not None:
                    h_tensor, _ = hidden
                    h = h_tensor[-1:, :, :].permute(1, 0, 2).squeeze(1)  # (1, hidden_dim)
                else:
                    h = torch.zeros(1, 512).to(device)
                
                # Get action
                action_tensor = control_model(z=z, h=h)  # (1, 3)
                prev_action = action_tensor.clone()
                action = action_tensor.cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    env.close()
    return np.mean(total_rewards)


dataset_path = 'datasets'
model_path = 'trained_models'
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained V and M models
    print("Loading pretrained models...")
    vision_model = VAE(latent_dim=32)
    memory_model = MDNRNN(latent_dim=32, action_dim=3, hidden_size=512, num_layers=1, num_mixtures=5)
    vision_model.load_state_dict(torch.load(model_path + '/vision_model.pth', map_location=device))
    memory_model.load_state_dict(torch.load(model_path + '/memory_model.pth', map_location=device))
    vision_model.to(device)
    memory_model.to(device)
    vision_model.eval()
    memory_model.eval()
    
    # Load initial frames for dream rollouts
    print("Loading initial frames...")
    frames = np.load(dataset_path + '/car_racing_recorded_frames.npy')
    initial_frames = np.transpose(frames[:1000], (0, 3, 1, 2))  # Use 1000 diverse starting points
    
    # Create world model environment (without reward model)
    world_env = WorldModelEnv(vision_model, memory_model, device)
    
    # Initialize controller - Use smaller architecture for CMA-ES
    print("Creating compact controller for evolution...")
    control_model = ControlModel(latent_dim=32, hidden_dim=512, output_dim=3)
    control_model.to(device)
    control_model.eval()
    
    # Get initial parameters
    initial_params = []
    for param in control_model.parameters():
        initial_params.extend(param.data.cpu().numpy().flatten())
    initial_params = np.array(initial_params)
    
    print(f"Controller has {len(initial_params)} parameters")
    
    # Use Sep-CMA-ES for large parameter spaces (diagonal covariance only)
    print("\nStarting Sep-CMA-ES optimization (memory efficient)...")
    print("Training controller in dream using world model...")
    
    # CMA-ES settings
    population_size = 32  # Larger population for better exploration
    sigma0 = 0.1  # Smaller initial step size
    
    es = cma.CMAEvolutionStrategy(
        initial_params, 
        sigma0, 
        {
            'popsize': population_size, 
            'seed': 42,
            'CMA_diagonal': True,  # Use diagonal covariance (memory efficient)
            'verb_filenameprefix': 'outcmaes',
            'verb_log': 0
        }
    )
    
    generation = 0
    best_reward = -float('inf')
    
    try:
        while not es.stop():
            generation += 1
            
            # Ask for candidate solutions
            solutions = es.ask()
            
            # Evaluate each solution in the dream
            fitness_list = []
            for sol in tqdm(solutions, desc=f"Gen {generation}", leave=False):
                # Load parameters into model
                idx = 0
                for param in control_model.parameters():
                    param_shape = param.shape
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        sol[idx:idx+param_size].reshape(param_shape)
                    ).float().to(device)
                    idx += param_size
                
                # Rollout in dream multiple times and average
                rewards = []
                for _ in range(5):  # 5 rollouts per candidate
                    reward = rollout_in_dream(control_model, world_env, initial_frames, num_steps=50)
                    rewards.append(reward)
                
                avg_reward = np.mean(rewards)
                fitness_list.append(-avg_reward)  # CMA-ES minimizes, we want to maximize reward
            
            # Update CMA-ES
            es.tell(solutions, fitness_list)
            
            # Log progress
            best_fitness = -es.result.fbest
            mean_fitness = -np.mean(fitness_list)
            
            print(f"Gen {generation} - Best: {best_fitness:.4f}, Mean: {mean_fitness:.4f}, Sigma: {es.sigma:.4f}")
            
            # Save best model every 10 generations
            if generation % 10 == 0:
                # Load best solution
                best_sol = es.result.xbest
                idx = 0
                for param in control_model.parameters():
                    param_shape = param.shape
                    param_size = param.numel()
                    param.data = torch.from_numpy(
                        best_sol[idx:idx+param_size].reshape(param_shape)
                    ).float().to(device)
                    idx += param_size
                
                # Evaluate in real environment
                real_reward = evaluate_in_real_env(control_model, vision_model, memory_model, device, num_episodes=3)
                print(f"  → Real env reward: {real_reward:.4f}")
                
                if real_reward > best_reward:
                    best_reward = real_reward
                    torch.save(control_model.state_dict(), model_path + '/control_model_cmaes.pth')
                    print(f"  ✓ Saved new best model! Real reward: {best_reward:.4f}")
            
            # Early stopping if converged
            if generation >= 200:
                print("Reached maximum generations (200)")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("\nTraining complete!")
    print(f"Best real environment reward: {best_reward:.4f}")
    print("Model saved to control_model_cmaes.pth")
