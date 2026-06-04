"""Derivative-aware (trajectory-slope) Huber loss for NBEATSx.

Standard HuberLoss scores each horizon step independently, so the trained 26-step
trajectory can have correct point values but a noisy / biased step-to-step slope.
The thermal-control scheme (paper Sec 5, Eq 7-8) consumes dT/dt:

    T_setpoint(t) = Tbar_pred(t) + 0.5*tau*dT/dt - 0.3 degC

so slope fidelity matters operationally. This loss adds a Huber penalty on the
ALONG-HORIZON first difference:

    L = Huber(T_hat - T) + lam * Huber( diff_H(T_hat) - diff_H(T) )

where diff_H is the first difference along the horizon axis (axis=1). lam=0
reduces exactly to the stock HuberLoss (bit-for-bit).

Grounded against neuralforecast 3.1.9. BasePointLoss is invoked via __call__ as
    self.loss(y=outsample_y, y_hat=output, y_insample=insample_y, mask=outsample_mask)
with y/y_hat/mask of shape [B, H, 1] (horizon on axis=1). See
.venv/.../neuralforecast/losses/pytorch.py (BasePointLoss, HuberLoss,
_weighted_mean) and common/_base_model.py.
"""

from typing import Union

import torch
import torch.nn.functional as F

from neuralforecast.losses.pytorch import BasePointLoss, _weighted_mean


class TrajHuberLoss(BasePointLoss):
    """Huber point loss + lam * Huber loss on the along-horizon first difference.

    Args:
        lam (float): weight on the slope (first-difference) term. lam=0 == HuberLoss.
        delta (float): Huber transition threshold, applied to BOTH terms.
        horizon_weight: optional per-step weight tensor of length H (point term).
    """

    def __init__(self, lam: float = 0.5, delta: float = 1.0, horizon_weight=None):
        # outputsize_multiplier / output_names MUST match HuberLoss, or NBEATSx
        # sizes the output head incorrectly.
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=1,
            output_names=[""],
        )
        self.lam = float(lam)
        self.delta = float(delta)

    def __call__(
        self,
        y: torch.Tensor,                               # [B, H, 1]
        y_hat: torch.Tensor,                           # [B, H, 1]
        y_insample: Union[torch.Tensor, None] = None,
        mask: Union[torch.Tensor, None] = None,        # [B, H, 1] or None
    ) -> torch.Tensor:
        # ---- point term: identical to stock HuberLoss ----
        point_losses = F.huber_loss(y, y_hat, reduction="none", delta=self.delta)
        point = _weighted_mean(
            losses=point_losses, weights=self._compute_weights(y=y, mask=mask)
        )

        if self.lam == 0.0:
            return point

        # ---- slope term: Huber on the along-horizon first difference ----
        dy = torch.diff(y, n=1, dim=1)                 # [B, H-1, 1]
        dy_hat = torch.diff(y_hat, n=1, dim=1)         # [B, H-1, 1]
        slope_losses = F.huber_loss(dy, dy_hat, reduction="none", delta=self.delta)

        # A difference step (k -> k+1) is valid only if BOTH endpoints are valid.
        # min of adjacent weights == logical AND for 0/1 masks, and degrades
        # gracefully if horizon_weight introduces fractional weights.
        w = self._compute_weights(y=y, mask=mask)      # [B, H, 1]
        slope_w = torch.minimum(w[:, 1:, :], w[:, :-1, :])  # [B, H-1, 1]
        slope = _weighted_mean(losses=slope_losses, weights=slope_w)

        return point + self.lam * slope
