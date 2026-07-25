"""Lightweight clustering + dimensionality reduction for style fingerprints.

Pure numpy (no scikit-learn) so it stays dependency-light. Used to map a
comedian's set onto its comedic "genome space": cluster bits by their six axes
and project to 2D for visualization.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def kmeans(
    x: np.ndarray, k: int, iters: int = 50, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means. Returns (labels, centroids) in the ORIGINAL feature space."""
    n = x.shape[0]
    k = max(1, min(k, n))
    xs = _standardize(x)

    rng = np.random.default_rng(seed)
    # k-means++ style seeding: first center random, rest far from chosen
    centers = [xs[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min(
            np.stack([np.sum((xs - c) ** 2, axis=1) for c in centers], axis=0),
            axis=0,
        )
        probs = d / (d.sum() + 1e-12)
        centers.append(xs[rng.choice(n, p=probs)])
    cs = np.stack(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.stack([np.sum((xs - c) ** 2, axis=1) for c in cs], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = xs[labels == j]
            if len(members):
                cs[j] = members.mean(axis=0)

    # centroids reported in the original (un-standardized) space for readability
    centroids = np.stack(
        [x[labels == j].mean(axis=0) if np.any(labels == j) else x.mean(axis=0)
         for j in range(k)]
    )
    return labels, centroids


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Project rows of x to 2D via SVD. Returns an (n, 2) array."""
    if x.shape[0] == 0:
        return np.zeros((0, 2))
    if x.shape[0] == 1:
        return np.zeros((1, 2))
    xc = x - x.mean(axis=0)
    # guard against zero-variance columns
    try:
        u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((x.shape[0], 2))
    comps = u[:, :2] * s[:2]
    if comps.shape[1] < 2:  # pad if only one component
        comps = np.pad(comps, ((0, 0), (0, 2 - comps.shape[1])))
    return comps


def suggest_k(n: int) -> int:
    """Heuristic cluster count for a set of ``n`` bits."""
    if n <= 3:
        return max(1, n)
    if n <= 8:
        return 2
    if n <= 15:
        return 3
    return 4
