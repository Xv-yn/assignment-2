"""T[i,j]=P(observed label j | clean label i); row posteriors use p @ T.

Forward CE: Patrini et al. (CVPR 2017).
GCE: Zhang & Sabuncu (NeurIPS 2018). The optional importance weighting below
is our explicit adaptation, not part of the original GCE algorithm.
"""

import torch
import torch.nn.functional as F


def forward_loss(logits, labels, transition):
    # logsumexp avoids log(softmax @ T) underflow. Zero entries are exact zeros.
    log_t = transition.log()
    noisy_log_probs = torch.logsumexp(F.log_softmax(logits, dim=1).unsqueeze(2)
                                    + log_t.unsqueeze(0), dim=1)
    return F.nll_loss(noisy_log_probs, labels)


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
