import torch
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_samples(n, frame_size, noise_std=0.1) -> torch.Tensor:
    samples = torch.zeros(n, frame_size)

    for i in range(n):
        v = random.uniform(-1, 1)
        a = random.uniform(-0.05, 0.05)
        x = torch.randn(1).item()
        samples[i][0] = x + torch.randn(1).item() * noise_std

        for j in range(1, frame_size):
            x = x + v * j + 0.5 * a * (j ** 2)
            noise = torch.randn(1).item() * noise_std
            samples[i][j] = x + noise

    return samples

def visualize_movement(o_t):
    """
    Visualize the particle movement like a video animation.
    
    Args:
        o_t: A single trajectory of shape (frame_size,) representing positions over time
    """
    positions = o_t.numpy()
    frame_size = len(positions)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the plot limits
    y_min, y_max = positions.min(), positions.max()
    y_range = y_max - y_min
    ax.set_xlim(-1, frame_size)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Initialize plot elements
    line, = ax.plot([], [], 'b-o', linewidth=2, markersize=8, label='Trajectory')
    particle, = ax.plot([], [], 'ro', markersize=15, label='Current Position')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title('Particle Movement Animation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    def init():
        line.set_data([], [])
        particle.set_data([], [])
        time_text.set_text('')
        return line, particle, time_text
    
    def animate(frame):
        # Update trajectory line (show path up to current frame)
        x_data = list(range(frame + 1))
        y_data = positions[:frame + 1]
        line.set_data(x_data, y_data)
        
        # Update particle position
        particle.set_data([frame], [positions[frame]])
        
        # Update time text
        time_text.set_text(f'Frame: {frame}/{frame_size-1}')
        
        return line, particle, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frame_size, interval=200, 
                                   blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim