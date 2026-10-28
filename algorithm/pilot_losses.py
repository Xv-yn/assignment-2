"""Original implementations of forward correction and untruncated GCE."""

import torch
from torch.nn import functional as F


KNOWN_TRANSITIONS = {
    "FashionMNIST0.3": [[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]],
    "FashionMNIST0.6": [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
}


def validate_transition(matrix: torch.Tensor) -> None:
    if (matrix.shape != (3, 3) or not torch.isfinite(matrix).all() or
            (matrix < 0).any() or
            not torch.allclose(matrix.sum(1), torch.ones(3, device=matrix.device, dtype=matrix.dtype))):
        raise ValueError("Expected finite nonnegative 3x3 T with clean rows summing to one")


def observed_log_probabilities(logits, transition):
    # T[i,j] = P(S=j | Y=i). Stay in log space, including exact zero T entries.
    return torch.logsumexp(logits.log_softmax(1).unsqueeze(2) + transition.log().unsqueeze(0), dim=1)


def per_example_loss(logits, labels, method, *, gamma=0.7, transition=None):
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
