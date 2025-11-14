# TOPIC 1: Applying World Model For Learning The Movement of a Particle in 1D axis

## Dataset
Generated using the formula:
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?bg=transparent&x_{t}=x_{t-1}+vt+\frac{1}{2}at^2" />
</p>
## Model Architecture
observation $o_t$ -> encoder -> transition -> decoder -> predict observation $\hat{o}_t$


The encoder and decoder are simple MLP with 64 hidden size. The transition model using GRU for capturing the relationship between each position in future easily.


