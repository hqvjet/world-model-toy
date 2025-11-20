import torch
import gymnasium as gym
import time
import numpy as np
import random
import pygame

import numpy as np

# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Car Racing - Use Arrow Keys to Control")

env = gym.make(
    "CarRacing-v3", 
    render_mode='human',  # Changed to 'human' to show the game window
    lap_complete_percent=0.95, 
    domain_randomize=False,
    continuous=True
)
obs, info = env.reset()
epi = 10
max_steps = 1000
frames = []
actions = []
rewards = []  # Add rewards tracking
path = 'datasets'

def get_action_from_keys():
    """Convert arrow key presses to action vector [steering, gas, brake] with continuous values"""
    keys = pygame.key.get_pressed()
    
    steering = 0.0
    gas = 0.0
    brake = 0.0
    
    # Continuous steering based on how long keys are held
    if keys[pygame.K_LEFT]:
        steering = -0.8  # Can adjust strength (0.0 to 1.0)
    if keys[pygame.K_RIGHT]:
        steering += 0.8  # Additive for simultaneous key presses
    
    # Continuous gas/brake
    if keys[pygame.K_UP]:
        gas = 0.8  # Can adjust acceleration strength
    if keys[pygame.K_DOWN]:
        brake = 0.8  # Can adjust brake strength
    
    # Clamp values to valid range
    steering = np.clip(steering, -1.0, 1.0)
    gas = np.clip(gas, 0.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    
    return np.array([steering, gas, brake], dtype=np.float32)

for _ in range(epi):
    obs, info = env.reset()
    for step in range(max_steps):
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                exit()
        
        print(f"Episode {_}, Step {step}")
        action = get_action_from_keys()
        print(f"Action taken: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        
        frames.append(obs)
        actions.append(action)
        rewards.append(reward)  # Save reward synchronized with frame and action

        if terminated or truncated:
            obs, info = env.reset()

# Save frames, actions, and rewards
np_frames = np.array(frames)
np_actions = np.array(actions)
np_rewards = np.array(rewards)
print(f"Recorded frames shape: {np_frames.shape}")
print(f"Recorded actions shape: {np_actions.shape}")
print(f"Recorded rewards shape: {np_rewards.shape}")
print(f"Reward stats - Mean: {np_rewards.mean():.4f}, Std: {np_rewards.std():.4f}, Min: {np_rewards.min():.4f}, Max: {np_rewards.max():.4f}")

np.save(path + '/car_racing_recorded_frames.npy', np_frames)
np.save(path + '/car_racing_recorded_actions.npy', np_actions)
np.save(path + '/car_racing_recorded_rewards.npy', np_rewards)

pygame.quit()
env.close()

# === Prepare Memory Dataset (merged from prepare_memory_dataset.py) ===
print("\n=== Preparing Memory Dataset ===")

num_epi = 20
T_SEQ = 100

# Reshape into episodes
frame_data = np_frames.reshape((num_epi, -1) + np_frames.shape[1:])
frame_data = frame_data.transpose((0, 1, 4, 2, 3))  # (num_epi, steps, channels, height, width)
action_data = np_actions.reshape((num_epi, -1) + np_actions.shape[1:])
reward_data = np_rewards.reshape((num_epi, -1))  # (num_epi, steps)

print(f"Episode frames shape: {frame_data.shape}")
print(f"Episode actions shape: {action_data.shape}")
print(f"Episode rewards shape: {reward_data.shape}")

# Split into sequences of T_SEQ
frame_data = frame_data.reshape((-1, T_SEQ) + frame_data.shape[2:])
action_data = action_data.reshape((-1, T_SEQ) + action_data.shape[2:])
reward_data = reward_data.reshape((-1, T_SEQ))

print(f"Sequence frames shape: {frame_data.shape}")
print(f"Sequence actions shape: {action_data.shape}")
print(f"Sequence rewards shape: {reward_data.shape}")

# Save memory dataset
np.save(path + '/car_racing_memory_dataset_frames.npy', frame_data)
np.save(path + '/car_racing_memory_dataset_actions.npy', action_data)
np.save(path + '/car_racing_memory_dataset_rewards.npy', reward_data)

print("\n✓ Data collection and preparation complete!")
print(f"Saved {len(frame_data)} sequences of {T_SEQ} timesteps each")