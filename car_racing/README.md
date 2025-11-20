# Car Racing World Model

Implementation of World Models (Ha & Schmidhuber, 2018) for the CarRacing-v3 environment.

## Architecture

**V (Vision Model)**: VAE with latent_dim=32  
**M (Memory Model)**: MDN-RNN with LSTM (hidden_size=512, num_mixtures=5)  
**C (Controller)**: Linear controller (single layer)

## Training Pipeline

### 1. Collect Data
```bash
python car_racing/recorded_frames.py
```
- Drive manually using arrow keys (←→ steer, ↑ gas, ↓ brake)
- Records 20 episodes × 500 steps = 10,000 frames
- Saves synchronized frames, actions, and rewards
- Automatically prepares sequences for memory training

**Output:**
- `car_racing_recorded_frames.npy` - Raw frames
- `car_racing_recorded_actions.npy` - Human actions
- `car_racing_recorded_rewards.npy` - Environment rewards
- `car_racing_memory_dataset_*.npy` - Prepared sequences (100 sequences × 100 steps)

### 2. Train Vision Model (V)
```bash
python car_racing/train_vision.py
```
- Learns to encode/decode observations to latent space
- Uses KL annealing over 100 epochs
- Target: Reconstruction loss < 500

**Output:** `vision_model.pth`

### 3. Train Memory Model (M)
```bash
python car_racing/train_memory.py
```
- Predicts next latent state from (current_latent, action)
- MDN-RNN with mixture of Gaussians
- Target: MDN loss < 3.0

**Output:** `memory_model.pth`

### 4. Train Controller (C)
```bash
python car_racing/train_controller_cmaes.py
```
- Evolution Strategies (CMA-ES) optimization
- Trains entirely in the "dream" (world model)
- Tests in real environment every 10 generations

**Output:** `control_model_cmaes.pth`

### 5. Test Agent
```bash
python car_racing/car_racing_gameplay_simulator.py
```
- Runs 10 episodes with trained controller
- Uses action smoothing (α=0.3) to reduce jitter
- Shows reward and step count

## Visualization Tools

**Vision Quality:**
```bash
python car_racing/visualize_vision_model.py
python car_racing/visualize_vision_video.py
```

**Memory Prediction:**
```bash
python car_racing/visualize_memory_prediction.py
```
- Dreams for 50 steps, syncs with real world, repeats
- Creates side-by-side comparison video

## File Structure

```
car_racing/
├── models/
│   ├── vae.py              # Vision model (V)
│   ├── mdn_rnn.py          # Memory model (M)
│   └── ffn.py              # Controller (C)
├── recorded_frames.py      # Data collection + preparation
├── train_vision.py         # Train V
├── train_memory.py         # Train M
├── train_controller_cmaes.py  # Train C with CMA-ES
├── car_racing_gameplay_simulator.py  # Test agent
└── visualize_*.py          # Visualization tools
```

## Key Hyperparameters

**Vision (VAE):**
- latent_dim: 32
- batch_size: 100
- lr: 1e-4
- KL annealing: 100 epochs

**Memory (MDN-RNN):**
- hidden_size: 512
- num_mixtures: 5
- batch_size: 32
- lr: 1e-3
- gradient clipping: 5.0

**Controller (CMA-ES):**
- population_size: 32
- sigma0: 0.1
- rollout_steps: 50
- max_generations: 200

## Tips

- Good human driving data is crucial (aim for reward > 800 per episode)
- VAE should converge to loss ~400 after 10 epochs
- Memory model loss should stabilize around 2-3
- Controller evolution is slow (~1-2 hours for 100 generations)
- Action smoothing prevents oscillations during gameplay
