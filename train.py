import time, os
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# Reload best checkpoint to continue from there
ckpt = torch.load(os.path.join(cfg.SAVE_DIR, 'model_best.pth'))
model.load_state_dict(ckpt['state_dict'])
start_epoch = ckpt['epoch'] + 1
best_ade    = ckpt['ade']
print(f'Resuming from epoch {start_epoch}, ADE={best_ade:.4f}')

# Lower LR + no AMP to prevent NaN
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, weight_decay=cfg.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.EPOCHS - start_epoch + 1, eta_min=1e-6)

criterion_traj = nn.SmoothL1Loss()

for epoch in range(start_epoch, cfg.EPOCHS+1):
    model.train()
    train_loss = 0.0
    t0 = time.time()
    skip_count = 0

    for obs, pred in train_loader:
        obs  = obs.to(cfg.DEVICE)
        pred = pred.to(cfg.DEVICE)

        optimizer.zero_grad()
        trajs, confs = model(obs)

        gt_exp    = pred.unsqueeze(1).expand_as(trajs)
        traj_loss = criterion_traj(trajs, gt_exp)

        with torch.no_grad():
            dist      = torch.norm(trajs - gt_exp, dim=-1).mean(dim=-1)
            best_mode = dist.argmin(dim=-1)
        conf_loss = F.nll_loss(torch.log(confs + 1e-8), best_mode)
        loss      = traj_loss + 0.1 * conf_loss

        # Skip NaN batches
        if torch.isnan(loss):
            skip_count += 1
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    avg_loss = train_loss / max(len(train_loader) - skip_count, 1)

    # Validate
    model.eval()
    val_ades, val_fdes = [], []
    with torch.no_grad():
        for obs, pred in val_loader:
            obs  = obs.to(cfg.DEVICE)
            pred = pred.to(cfg.DEVICE)
            trajs, confs = model(obs)
            val_ades.append(compute_ade(trajs, pred))
            val_fdes.append(compute_fde(trajs, pred))

    val_ade = sum(val_ades) / len(val_ades)
    val_fde = sum(val_fdes) / len(val_fdes)
    lr_now  = scheduler.get_last_lr()[0]

    flag = ''
    if val_ade < best_ade:
        best_ade = val_ade
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'ade': best_ade,
            'fde': val_fde
        }, os.path.join(cfg.SAVE_DIR, 'model_best.pth'))
        flag = '  ** SAVED **'

    elapsed = time.time() - t0
    print(f'Ep {epoch:03d}/{cfg.EPOCHS} | '
          f'loss={avg_loss:.4f} | '
          f'ADE={val_ade:.4f} | '
          f'FDE={val_fde:.4f} | '
          f'lr={lr_now:.2e} | '
          f'{elapsed:.0f}s{flag}')

print(f'\nBest ADE: {best_ade:.4f}')
