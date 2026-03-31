import torch

def compute_ade(pred_trajs, gt_traj):
    '''
    pred_trajs : (B, num_modes, pred_len, 2)
    gt_traj    : (B, pred_len, 2)
    Returns best-of-K ADE (lower is better)
    '''
    gt = gt_traj.unsqueeze(1).expand_as(pred_trajs)
    dist = torch.norm(pred_trajs - gt, dim=-1)  # (B, K, pred_len)
    ade_per_mode = dist.mean(dim=-1)             # (B, K)
    best_ade     = ade_per_mode.min(dim=-1)[0]   # (B,)
    return best_ade.mean().item()

def compute_fde(pred_trajs, gt_traj):
    '''
    Returns best-of-K FDE (lower is better)
    '''
    gt_final = gt_traj[:, -1, :]                 # (B, 2)
    pred_final = pred_trajs[:, :, -1, :]         # (B, K, 2)
    dist = torch.norm(pred_final -
           gt_final.unsqueeze(1), dim=-1)        # (B, K)
    best_fde = dist.min(dim=-1)[0]               # (B,)
    return best_fde.mean().item()

print('ADE/FDE metrics ready.')
