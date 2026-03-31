# Intent & Trajectory Prediction — MaHe Mobility Hackathon

## Problem Statement
Predicting future coordinates (next 3 seconds) of pedestrians 
and cyclists based on 2 seconds of past motion in an L4 urban environment.

## Model Results
| Metric | Score |
|--------|-------|
| ADE | 0.3127 m |
| FDE | 0.5837 m |
| Best Epoch | 46 |

## Dataset
Trained on the nuScenes dataset.
Download from: https://www.nuscenes.org/

## How to Run
1. Download nuScenes dataset
2. Open the notebook
3. Run all cells

## Tech Stack
- Python
- PyTorch
- nuScenes
