"""Known noise matrices and training-partition-only anchor estimation."""

import numpy as np

KNOWN = {
    "FashionMNIST0.3": np.array([[.7, .3, 0], [0, .7, .3], [.3, 0, .7]]),
    "FashionMNIST0.6": np.array([[.4, .3, .3], [.3, .4, .3], [.3, .3, .4]]),
}


def validate_transition(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Transition must be a finite nonnegative 3x3 matrix")
    if not np.allclose(matrix.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Transition rows must sum to one")
    return matrix


def estimate_anchor(probabilities, top_fraction=0.01, quantile=1.0):
    """Average the most confident noisy posteriors for each class.

    With fraction=0 (or a single selected point), this is the original argmax
    anchor estimator. Optional quantile<1 excludes extreme confidence outliers.
    Assumes anchor examples exist and class IDs are aligned (typically diagonal
    dominance). Neither assumption is identifiable from noisy labels alone.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not len(probabilities):
        raise ValueError("Expected a nonempty N x 3 posterior array")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid posterior probabilities")
    if not 0 <= top_fraction <= 1 or not 0 < quantile <= 1:
        raise ValueError("Invalid anchor fraction/quantile")
    rows, selected = [], []
    for label in range(3):
        scores = probabilities[:, label]
        candidates = np.flatnonzero(scores <= np.quantile(scores, quantile))
        count = max(1, int(np.ceil(len(candidates) * top_fraction)))
        indices = candidates[np.argsort(-scores[candidates], kind="stable")[:count]]
        rows.append(probabilities[indices].mean(0))
        selected.append(indices.tolist())
    matrix = validate_transition(np.stack(rows))
    condition = float(np.linalg.cond(matrix))
    return matrix, {"anchor_local_indices": selected,
                    "top_fraction": top_fraction, "quantile": quantile,
                    "condition_number": condition if np.isfinite(condition) else None,
                    "diagonal_dominant": bool(np.all(matrix.diagonal() >= matrix.max(1))),
                    "assumption": "anchor points and aligned clean/noisy class IDs"}
