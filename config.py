import os
from dataclasses import dataclass

@dataclass
class CFG:
    DATAROOT     : str   = '/kaggle/working/nuscenes/'
    VERSION      : str   = 'v1.0-trainval'
    SAVE_DIR     : str   = '/kaggle/working/checkpoints'
    # Trajectory settings
    OBS_SECS     : float = 2.0   # observe 2 seconds of past
    PRED_SECS    : float = 3.0   # predict 3 seconds future
    FREQ         : float = 2.0   # nuScenes annotation frequency (2Hz)
    OBS_LEN      : int   = 4     # 2s x 2Hz = 4 past positions
    PRED_LEN     : int   = 6     # 3s x 2Hz = 6 future positions
    NUM_MODES    : int   = 3     # predict 3 possible paths
    # Model
    D_MODEL      : int   = 128
    N_HEADS      : int   = 8
    N_LAYERS     : int   = 4
    DROPOUT      : float = 0.1
    MAX_AGENTS   : int   = 20    # max neighbours per scene
    # Training
    EPOCHS       : int   = 60
    BATCH        : int   = 64
    LR           : float = 1e-3
    WD           : float = 1e-4
    SEED         : int   = 42
    DEVICE       : str   = 'cuda'
    NUM_WORKERS  : int   = 2

cfg = CFG()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
print('Config loaded.')
print(f'Obs steps : {cfg.OBS_LEN} | Pred steps : {cfg.PRED_LEN}')
