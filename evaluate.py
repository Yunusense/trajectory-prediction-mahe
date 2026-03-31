import torch, os
from torch.amp import autocast

ckpt = torch.load(os.path.join(cfg.SAVE_DIR, 'model_best.pth'))
model.load_state_dict(ckpt['state_dict'])
print(f'Loaded best model from epoch {ckpt["epoch"]}')

model.eval()
test_ades, test_fdes = [], []
with torch.no_grad():
    for obs, pred in test_loader:
        obs  = obs.to(cfg.DEVICE)
        pred = pred.to(cfg.DEVICE)
        trajs, confs = model(obs)
        test_ades.append(compute_ade(trajs, pred))
        test_fdes.append(compute_fde(trajs, pred))

test_ade = sum(test_ades) / len(test_ades)
test_fde = sum(test_fdes) / len(test_fdes)

print('\n=== FINAL TEST RESULTS ===')
print(f'ADE : {test_ade:.4f} m')
print(f'FDE : {test_fde:.4f} m')
print(f'Best epoch : {ckpt["epoch"]}')
