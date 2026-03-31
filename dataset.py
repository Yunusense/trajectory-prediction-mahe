import numpy as np
from nuscenes.nuscenes import NuScenes

def extract_trajectories(nusc, obs_len=4, pred_len=6):
    '''
    Extract pedestrian and cyclist trajectories from nuScenes.
    Returns list of dicts with obs (past) and pred (future) positions.
    Categories: pedestrian, bicycle, motorcycle
    '''
    TARGET_CATS = {'human.pedestrian.adult',
                   'human.pedestrian.child',
                   'human.pedestrian.wheelchair',
                   'vehicle.bicycle',
                   'vehicle.motorcycle'}

    # Build per-instance timeline: instance_token -> list of (sample_idx, x, y)
    print('Building instance timelines...')
    instance_timeline = {}

    for ann in nusc.sample_annotation:
        inst_token = ann['instance_token']
        cat = nusc.get('category',
              nusc.get('instance', inst_token)['category_token'])['name']
        if not any(cat.startswith(t) for t in TARGET_CATS):
            continue
        sample = nusc.get('sample', ann['sample_token'])
        x, y   = ann['translation'][0], ann['translation'][1]
        if inst_token not in instance_timeline:
            instance_timeline[inst_token] = []
        instance_timeline[inst_token].append(
            (sample['timestamp'], x, y, cat))

    # Sort each instance by timestamp
    for k in instance_timeline:
        instance_timeline[k].sort(key=lambda t: t[0])

    # Slide window: obs_len past + pred_len future
    print('Extracting sequences...')
    sequences = []
    total_len = obs_len + pred_len

    for inst_token, timeline in instance_timeline.items():
        if len(timeline) < total_len:
            continue
        for i in range(len(timeline) - total_len + 1):
            window = timeline[i:i+total_len]
            obs  = np.array([[w[1], w[2]] for w in window[:obs_len]],
                            dtype=np.float32)
            pred = np.array([[w[1], w[2]] for w in window[obs_len:]],
                            dtype=np.float32)
            sequences.append({
                'obs':       obs,       # (obs_len, 2)
                'pred':      pred,      # (pred_len, 2)
                'category':  window[0][3],
                'inst_token': inst_token
            })

    print(f'Total sequences: {len(sequences)}')
    return sequences

sequences = extract_trajectories(nusc, cfg.OBS_LEN, cfg.PRED_LEN)
print(f'Sample obs  shape: {sequences[0]["obs"].shape}')
print(f'Sample pred shape: {sequences[0]["pred"].shape}')

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, max_agents=20):
        self.sequences  = sequences
        self.max_agents = max_agents
        # Group sequences by instance for social context
        self.inst_map = {}
        for i, s in enumerate(sequences):
            self.inst_map.setdefault(s['inst_token'], []).append(i)

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        obs  = torch.tensor(seq['obs'],  dtype=torch.float32)  # (obs_len,2)
        pred = torch.tensor(seq['pred'], dtype=torch.float32)  # (pred_len,2)

        # Normalise: centre on last observed position
        origin = obs[-1].clone()
        obs    = obs  - origin
        pred   = pred - origin

        return obs, pred

# Split 80/10/10
n       = len(sequences)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)
n_test  = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(
    TrajectoryDataset(sequences),
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(cfg.SEED)
)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH,
    shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH,
    shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH,
    shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
print('DataLoaders ready.')
