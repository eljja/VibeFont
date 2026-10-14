"""Evaluation metrics for handwriting: CER/WER (readability) and DTW diversity."""

from __future__ import annotations

import numpy as np


def levenshtein_distance(seq1: list | str, seq2: list | str) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[m][n]


def cer(pred: str, target: str) -> float:
    """Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, target) / len(target)


def wer(pred: str, target: str) -> float:
    """Word Error Rate."""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(pred_words, target_words) / len(target_words)


def dtw_distance(xy1: np.ndarray, xy2: np.ndarray) -> float:
    """Dynamic Time Warping distance between two 2D point trajectories.

    Used to measure instance diversity across repeated generations of the same text.
    """
    n, m = len(xy1), len(xy2)
    if n == 0 or m == 0:
        return 0.0
    cost = np.linalg.norm(xy1[:, None, :] - xy2[None, :, :], axis=-1)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / max(n, m))
