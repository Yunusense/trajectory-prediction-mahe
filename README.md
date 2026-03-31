# Trajectory Prediction
## PS1 — Intent & Trajectory Prediction | MIT Bengaluru Hackathon
A multi-modal Social Transformer that predicts the future trajectories
of pedestrians and cyclists in urban autonomous driving scenarios.
## Problem Statement
Given 2 seconds of past (x,y) motion, predict the next 3 seconds
of future positions for pedestrians and cyclists.
## Model Architecture
- **Encoder**: 4-layer Transformer with positional encoding
- **Decoder**: Multi-modal decoder with 3 learnable mode queries
- **Input**: 4 past positions (2s at 2Hz) per agent
- **Output**: 3 possible future trajectories × 6 positions (3s at 2Hz)
- **Parameters**: ~2M (trained from scratch)
## Results
| Metric | Value | Description |
|--------|-------|-------------|
| ADE | 0.3106 m | Average Displacement Error (lower is better) |
| FDE | 0.5734 m | Final Displacement Error (lower is better) |
## Dataset
- nuScenes trainval — 850 scenes, ~60,000 trajectories
- Agents: pedestrians + cyclists
- No pretrained weights used
## How to Run
```bash
pip install nuscenes-devkit numpy torch
python train.py
python evaluate.py
```
## Key Features
- Multi-modal prediction (3 most likely paths)
- Social context via self-attention
- Velocity-aware input embedding
- Best-of-K evaluation (minADE@3, minFDE@3)
```


