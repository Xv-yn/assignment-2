"""T[i,j]=P(observed label j | clean label i); row posteriors use p @ T.

Forward CE: Patrini et al. (CVPR 2017).
GCE: Zhang & Sabuncu (NeurIPS 2018). The optional importance weighting below
is our explicit adaptation, not part of the original GCE algorithm.
"""

import torch
import torch.nn.functional as F

from .transition import KNOWN

KNOWN_TRANSITIONS = {name: matrix.tolist() for name, matrix in KNOWN.items()}


def observed_log_probabilities(logits, transition):
    # logsumexp avoids log(softmax @ T) underflow. Zero entries are exact zeros.
    log_t = transition.log()
    return torch.logsumexp(F.log_softmax(logits, dim=1).unsqueeze(2)
                          + log_t.unsqueeze(0), dim=1)


def forward_loss(logits, labels, transition):
    return F.nll_loss(observed_log_probabilities(logits, transition), labels)


def validate_transition_tensor(matrix):
    """Preserve the pilot tensor validator's device/dtype and tolerance."""
    if (matrix.shape != (3, 3) or not torch.isfinite(matrix).all() or
            (matrix < 0).any() or
            not torch.allclose(matrix.sum(1), torch.ones(3, device=matrix.device, dtype=matrix.dtype))):
        raise ValueError("Expected finite nonnegative 3x3 T with clean rows summing to one")


def per_example_loss(logits, labels, method, *, gamma=0.7, transition=None):
    """Original unweighted pilot objectives, including stable expm1 GCE.

    Formal weighted GCE below retains its original clamping and weighting.
    """
    if method == "ce":
        return F.cross_entropy(logits, labels, reduction="none")
    if method == "gce":
        if not 0 < gamma <= 1:
            raise ValueError("GCE gamma must lie in (0, 1]")
        log_p = logits.log_softmax(1).gather(1, labels[:, None]).squeeze(1)
        return -torch.expm1(gamma * log_p) / gamma
    if method == "forward":
        if transition is None:
            raise ValueError("Forward correction needs a supplied or estimated transition matrix")
        return F.nll_loss(observed_log_probabilities(logits, transition), labels, reduction="none")
    raise ValueError(f"Unknown method: {method}")


def gce_loss(logits, labels, q=0.7, transition=None, reweight=False, max_weight=10.0):
    if not 0 < q <= 1:
        raise ValueError("GCE q must lie in (0, 1]")
    probabilities = logits.softmax(dim=1)
    chosen = probabilities.gather(1, labels[:, None]).squeeze(1)
    losses = (1 - chosen.clamp_min(1e-12).pow(q)) / q
    if reweight:
        if transition is None:
            raise ValueError("Transition matrix required for importance reweighting")
        noisy_chosen = (probabilities @ transition).gather(1, labels[:, None]).squeeze(1)
        # A fixed plug-in estimate of p(Y=s|x)/p(S=s|x), not differentiated.
        weights = (chosen / noisy_chosen.clamp_min(1e-8)).detach().clamp(max=max_weight)
        losses = losses * weights
    return losses.mean()
