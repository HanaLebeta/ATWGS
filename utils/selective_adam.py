import torch

@torch.no_grad()
def selective_adam_step(optimizer, visibility_mask):
    N = visibility_mask.shape[0]
    inv_mask = ~visibility_mask
    saved = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            if p.dim() >= 1 and p.shape[0] == N:
                state = optimizer.state.get(p)
                if state and 'exp_avg' in state:
                    saved[p] = (state['exp_avg'][inv_mask].clone(), state['exp_avg_sq'][inv_mask].clone(), p.data[inv_mask].clone())
                p.grad.data[inv_mask] = 0
    optimizer.step()
    for p, (ea, eas, pval) in saved.items():
        state = optimizer.state[p]
        state['exp_avg'][inv_mask] = ea
        state['exp_avg_sq'][inv_mask] = eas
        p.data[inv_mask] = pval
