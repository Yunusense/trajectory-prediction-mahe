import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class SocialTransformer(nn.Module):
    '''
    Social Transformer for multi-modal trajectory prediction.
    Input  : obs trajectory (obs_len, 2)
    Output : K predicted trajectories (num_modes, pred_len, 2)
             + K confidence scores
    '''
    def __init__(self, obs_len=4, pred_len=6, d_model=128,
                 n_heads=8, n_layers=4, num_modes=3, dropout=0.1):
        super().__init__()
        self.obs_len   = obs_len
        self.pred_len  = pred_len
        self.d_model   = d_model
        self.num_modes = num_modes

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Velocity embedding
        self.vel_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder (social context)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=n_layers)

        # Multi-modal decoder heads
        self.mode_queries = nn.Parameter(
            torch.randn(num_modes, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer,
                                              num_layers=n_layers)

        # Output heads
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, pred_len * 2)
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        '''
        obs: (B, obs_len, 2)
        returns:
            trajs: (B, num_modes, pred_len, 2)
            confs: (B, num_modes)
        '''
        B = obs.size(0)

        # Compute velocities
        vel = torch.zeros_like(obs)
        vel[:, 1:] = obs[:, 1:] - obs[:, :-1]

        # Embed positions + velocities
        pos_emb = self.input_embed(obs)   # (B, obs_len, d)
        vel_emb = self.vel_embed(vel)     # (B, obs_len, d)
        x = pos_emb + vel_emb

        # Add positional encoding
        x = self.pos_enc(x)

        # Encode with transformer
        memory = self.transformer(x)     # (B, obs_len, d)

        # Multi-modal decoding
        queries = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        # (B, num_modes, d)

        decoded = self.decoder(queries, memory)  # (B, num_modes, d)

        # Predict trajectories and confidences
        trajs = self.traj_head(decoded)          # (B, num_modes, pred_len*2)
        trajs = trajs.view(B, self.num_modes,
                           self.pred_len, 2)     # (B, num_modes, pred_len, 2)
        confs = self.conf_head(decoded).squeeze(-1)  # (B, num_modes)
        confs = F.softmax(confs, dim=-1)

        return trajs, confs

model = SocialTransformer(
    obs_len   = cfg.OBS_LEN,
    pred_len  = cfg.PRED_LEN,
    d_model   = cfg.D_MODEL,
    n_heads   = cfg.N_HEADS,
    n_layers  = cfg.N_LAYERS,
    num_modes = cfg.NUM_MODES,
    dropout   = cfg.DROPOUT
).to(cfg.DEVICE)

total = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model parameters: {total:.2f}M')
print('Social Transformer ready.')
