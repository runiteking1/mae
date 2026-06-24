
import math
import torch

#############################
# From https://github.com/MoonshotAI/Moonlight
#############################


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


#############################
# Polar Express (Amsel et al., 2025, arXiv:2505.16932)
#############################

# Per-iteration coefficients for the odd polynomial p(x) = a*x + b*x^3 + c*x^5.
# Unlike vanilla Muon's single fixed triple repeated `ns_steps` times, Polar Express
# solves a minimax problem per iteration so orthogonalization converges faster/more
# accurately; the schedule converges to the optimal asymptotic (1.875, -1.25, 0.375).
# The /1.01^k factors are the polynomial safety factor that contracts each iterate to
# stay convergent under round-off (the last, exact, iterate needs none).
_POLAR_EXPRESS_COEFFS = [
    (8.28721201814563   / 1.01, -23.595886519098837 / (1.01**3), 17.300387312530933  / (1.01**5)),
    (4.107059111542203  / 1.01,  -2.9478499167379106 / (1.01**3),  0.5448431082926601 / (1.01**5)),
    (3.9486908534822946 / 1.01,  -2.908902115962949  / (1.01**3),  0.5518191394370137 / (1.01**5)),
    (3.3184196573706015 / 1.01,  -2.488488024314874  / (1.01**3),  0.51004894012372   / (1.01**5)),
    (2.300652019954817  / 1.01,  -1.6689039845747493 / (1.01**3),  0.4188073119525673 / (1.01**5)),
    (1.891301407787398  / 1.01,  -1.2679958271945868 / (1.01**3),  0.37680408948524835/ (1.01**5)),
    (1.8750014808534479 / 1.01,  -1.2500016453999487 / (1.01**3),  0.3750001645474248 / (1.01**5)),
    (1.875,                       -1.25,                            0.375),
]


@torch.compile
def zeropower_polar_express(G):
    """
    Polar Express orthogonalization: same X <- a*X + b*(X X^T) X + c*(X X^T)^2 X update as
    Newton-Schulz, but with the iteration-dependent coefficient schedule above (a fixed 8 steps,
    so there is no `steps` argument). The input is normalized so its spectral norm is < 1 (an
    extra /1.01 input-safety margin), and compute is in float32 for stability.
    """
    assert G.ndim == 2
    transposed = G.shape[0] > G.shape[1]
    X = (G.mT if transposed else G).float()
    X = X / (X.norm(p="fro") + 1e-2)
    X = X / 1.01
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.mT if transposed else X).to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        params=None,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        polar=False,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            polar=polar,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        if muon_params is None and adamw_params is None:
            # `params` is an iterable of group dicts; each must declare `use_muon`.
            # Lets callers pass per-layer groups carrying `lr_scale` / `weight_decay`,
            # which `step()` already honors via `self.param_groups`.
            super().__init__(params, defaults)
            for group in self.param_groups:
                use_muon = group["use_muon"]
                for p in group["params"]:
                    if use_muon:
                        assert p.ndim == 2, p.ndim
                    self.state[p]["use_muon"] = use_muon
        else:
            muon_params = list(muon_params) if muon_params is not None else []
            adamw_params = list(adamw_params) if adamw_params is not None else []
            all_params = muon_params + list(adamw_params)
            super().__init__(all_params, defaults)
            for p in muon_params:
                assert p.ndim == 2, p.ndim
                self.state[p]["use_muon"] = True
            for p in adamw_params:
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if group["polar"]:
                    u = zeropower_polar_express(g)
                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                p.data.mul_(1 - lr * (0.0 if p.ndim == 1 else weight_decay))
                p.data.add_(g, alpha=-lr / scale)

        return loss

#############################
