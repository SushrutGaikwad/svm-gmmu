"""Input validation utilities for SVM-GMU.

This module provides validation for the ``sample_uncertainty`` parameter
passed to :meth:`SvmGmu.fit`.  Each function raises ``ValueError`` with
a descriptive message when an invariant is violated, so the user gets a
clear explanation rather than a cryptic NumPy error from deep inside the
math routines.

Public functions
----------------
validate_sample_uncertainty
    Full validation of the list-of-dicts structure.
build_default_uncertainty
    Creates trivial (zero-variance, single-component) uncertainty entries
    so that the math code can always assume uncertainty is present.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Tolerance for checking that mixture weights sum to 1.
_WEIGHT_TOL = 1e-6


def validate_sample_uncertainty(
    sample_uncertainty: list[dict],
    n_samples: int,
    n_features: int,
) -> list[dict]:
    """Validate and normalize the sample_uncertainty structure.

    Checks performed (in order):

    1. ``sample_uncertainty`` is a list of length ``n_samples``.
    2. Each element is a dict with keys "weights", "means", "covariances".
    3. All arrays are converted to float64 numpy arrays.
    4. For each sample *i* with *M_i* components:
       - weights has shape (M_i,), all >= 0, and sums to 1.
       - means has shape (M_i, n_features).
       - covariances has shape (M_i, n_features) for diagonal
         or (M_i, n_features, n_features) for full.
       - Diagonal covariance entries are all >= 0.
    5. Component counts are consistent within each sample.

    Parameters
    ----------
    sample_uncertainty : list of dict
        The raw user-provided uncertainty data.
    n_samples : int
        Expected number of samples (must match len(sample_uncertainty)).
    n_features : int
        Dimensionality *d* (must match the second axis of means).

    Returns
    -------
    list of dict
        The validated and numpy-converted structure.  Each dict has
        keys "weights", "means", "covariances" with proper dtypes.

    Raises
    ------
    TypeError
        If the top-level object or an element is the wrong type.
    ValueError
        If any shape, value, or consistency check fails.
    """
    # -- Top-level type check -------------------------------------------
    if not isinstance(sample_uncertainty, list):
        raise TypeError(
            f"sample_uncertainty must be a list, got {type(sample_uncertainty).__name__}."
        )
    if len(sample_uncertainty) != n_samples:
        raise ValueError(
            f"sample_uncertainty has {len(sample_uncertainty)} entries but "
            f"X has {n_samples} samples.  They must match."
        )

    required_keys = {"weights", "means", "covariances"}
    validated: list[dict] = []

    for i, entry in enumerate(sample_uncertainty):
        prefix = f"sample_uncertainty[{i}]"

        # -- Dict structure check ---------------------------------------
        if not isinstance(entry, dict):
            raise TypeError(f"{prefix} must be a dict, got {type(entry).__name__}.")
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(f"{prefix} is missing keys: {missing}.")

        # -- Convert to numpy -------------------------------------------
        weights = np.asarray(entry["weights"], dtype=np.float64)
        means = np.asarray(entry["means"], dtype=np.float64)
        covs = np.asarray(entry["covariances"], dtype=np.float64)

        # -- Weights ----------------------------------------------------
        if weights.ndim != 1:
            raise ValueError(
                f"{prefix}['weights'] must be 1-D, got shape {weights.shape}."
            )
        m_i = weights.shape[0]
        if m_i == 0:
            raise ValueError(f"{prefix} must have at least one component.")
        if np.any(weights < 0.0):
            raise ValueError(f"{prefix}['weights'] contains negative values.")
        weight_sum = weights.sum()
        if abs(weight_sum - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                f"{prefix}['weights'] sum to {weight_sum:.8f}, expected 1.0 "
                f"(tolerance {_WEIGHT_TOL})."
            )
        # Normalize to exactly 1.0 to avoid drift in loss computation.
        weights = weights / weight_sum

        # -- Means ------------------------------------------------------
        if means.ndim != 2:
            raise ValueError(
                f"{prefix}['means'] must be 2-D (M_i, d), got shape {means.shape}."
            )
        if means.shape[0] != m_i:
            raise ValueError(
                f"{prefix}['means'] has {means.shape[0]} rows but "
                f"'weights' has {m_i} components."
            )
        if means.shape[1] != n_features:
            raise ValueError(
                f"{prefix}['means'] has {means.shape[1]} features but "
                f"X has {n_features} features."
            )

        # -- Covariances ------------------------------------------------
        _validate_covariances(covs, m_i, n_features, prefix)

        validated.append(
            {
                "weights": weights,
                "means": means,
                "covariances": covs,
            }
        )

    return validated


def _validate_covariances(
    covs: NDArray[np.floating],
    m_i: int,
    n_features: int,
    prefix: str,
) -> None:
    """Validate covariance array for one sample.

    Accepts either diagonal covariances of shape (M_i, d) or full
    covariance matrices of shape (M_i, d, d).  The format is
    auto-detected from the number of dimensions.

    Parameters
    ----------
    covs : ndarray
        The covariance array to validate.
    m_i : int
        Expected number of components.
    n_features : int
        Expected dimensionality.
    prefix : str
        Label for error messages (e.g. "sample_uncertainty[3]").

    Raises
    ------
    ValueError
        If the shape or values are invalid.
    """
    if covs.ndim == 2:
        # Diagonal covariances: shape (M_i, d)
        if covs.shape != (m_i, n_features):
            raise ValueError(
                f"{prefix}['covariances'] has shape {covs.shape}, expected "
                f"({m_i}, {n_features}) for diagonal covariances."
            )
        if np.any(covs < 0.0):
            raise ValueError(
                f"{prefix}['covariances'] contains negative diagonal entries."
            )
    elif covs.ndim == 3:
        # Full covariance matrices: shape (M_i, d, d)
        if covs.shape != (m_i, n_features, n_features):
            raise ValueError(
                f"{prefix}['covariances'] has shape {covs.shape}, expected "
                f"({m_i}, {n_features}, {n_features}) for full covariances."
            )
        # Symmetry and positive semi-definiteness check per component.
        for m in range(m_i):
            mat = covs[m]
            if not np.allclose(mat, mat.T, atol=1e-8):
                raise ValueError(f"{prefix}['covariances'][{m}] is not symmetric.")
            eigvals = np.linalg.eigvalsh(mat)
            if np.any(eigvals < -1e-8):
                raise ValueError(
                    f"{prefix}['covariances'][{m}] is not positive "
                    f"semi-definite (min eigenvalue = {eigvals.min():.2e})."
                )
    else:
        raise ValueError(
            f"{prefix}['covariances'] must be 2-D (diagonal) or 3-D (full), "
            f"got {covs.ndim}-D with shape {covs.shape}."
        )


def build_default_uncertainty(
    X: NDArray[np.floating],
) -> list[dict]:
    """Create trivial uncertainty entries (zero-variance point masses).

    Used when ``sample_uncertainty=None`` is passed to :meth:`SvmGmu.fit`,
    so the math code can uniformly assume uncertainty data is present.
    Each sample gets a single component with weight 1, mean equal to the
    row of X, and a zero diagonal covariance.  This makes the loss reduce
    to the standard hinge loss (Section 9.7 of the report).

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix whose rows are the sample points.

    Returns
    -------
    list of dict
        One entry per sample with trivial single-component GMMs.
    """
    n, d = X.shape
    return [
        {
            "weights": np.array([1.0]),
            "means": X[i : i + 1].copy(),  # shape (1, d)
            "covariances": np.zeros((1, d)),  # diagonal zeros, shape (1, d)
        }
        for i in range(n)
    ]


def validate_labels(y: NDArray) -> NDArray[np.float64]:
    """Ensure labels are in {+1, -1} and return as float64.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Raw class labels.

    Returns
    -------
    ndarray of shape (n,)
        Labels as float64, validated to contain only +1 and -1.

    Raises
    ------
    ValueError
        If any label is not +1 or -1.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    unique = set(np.unique(y))
    if not unique.issubset({1.0, -1.0}):
        raise ValueError(
            f"Labels must be in {{+1, -1}}, got unique values: {sorted(unique)}."
        )
    return y
