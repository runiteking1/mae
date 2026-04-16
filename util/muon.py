"""
Muon optimizer — Newton-Schulz orthogonalized momentum for 2D+ weight matrices.

Based on Keller Jordan's implementation from the modded-nanogpt project.
For parameters with ndim < 2 (biases, norms, embeddings), falls back to AdamW.
"""

import torch
from torch.optim import Optimizer


def newton_schulz_(G, steps=5, eps=1e-7):
    """
    Apply Newton-Schulz iterations to compute the orthogonal component of G.
    This approximates G @ (G^T G)^{-1/2}.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()

    # Reshape to 2D for the iteration
    orig_shape = X.shape
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # Transpose so rows <= cols for numerical stability
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype).reshape(orig_shape)


class Muon(Optimizer):
    """
    Muon optimizer: applies Newton-Schulz orthogonalization to momentum for
    2D+ weight parameters. 1D parameters (biases, norms) use AdamW-style updates.

    Args:
        muon_params: parameters to apply Muon update (should be 2D+ weights)
        lr: learning rate for Muon params (default: 0.02)
        momentum: momentum factor (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration count (default: 5)
        adamw_params: parameters to update with AdamW (biases, norms, embeddings)
        adamw_lr: learning rate for AdamW params (default: 1e-3)
        adamw_betas: AdamW beta coefficients (default: (0.9, 0.95))
        adamw_wd: AdamW weight decay (default: 0.05)
    """

    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_params=None, adamw_lr=1e-3,
                 adamw_betas=(0.9, 0.95), adamw_wd=0.05):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)

        params = list(muon_params)
        param_groups = [{"params": params, "type": "muon"}]

        if adamw_params is not None:
            adamw_list = list(adamw_params)
            if adamw_list:
                param_groups.append({
                    "params": adamw_list,
                    "type": "adamw",
                    "lr": adamw_lr,
                    "betas": adamw_betas,
                    "weight_decay": adamw_wd,
                })

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("type") == "adamw":
                self._adamw_step(group)
            else:
                self._muon_step(group)

        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)

            if nesterov:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf.clone()

            if g.ndim >= 2:
                g = newton_schulz_(g, steps=ns_steps)

            p.add_(g, alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)

            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            # weight decay
            p.mul_(1 - lr * wd)

            # Adam update
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(1e-8)
            step_size = lr / bias_correction1

            p.addcdiv_(exp_avg, denom, value=-step_size)
