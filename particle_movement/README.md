# TOPIC 1: Applying World Model For Learning The Movement of a Particle in 1D axis

## Dataset
Generated using the formula:
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?bg=transparent&x_%7Bt%7D%20%3D%20x_%7Bt-1%7D%20%2B%20vt%20%2B%20%5Cfrac%7B1%7D%7B2%7D%20at%5E2" />
</p>

## Model Architecture
observation $o_t$ -> encoder -> transition -> decoder -> predict observation $\hat{o}_t$


The encoder and decoder are simple MLP with 64 hidden size. The transition model using GRU for capturing the relationship between each position in future easily.



