"""Closed-form loss and gradient computations for SVM-GMMU.

Every public function in this module corresponds to one or more equations
from the paper.  The mapping is:

    compute_d_mu        -> Eq. 13 / 25   (signed margin distance of mean)
    compute_d_sigma     -> Eq. 14 / 26   (uncertainty spread along w)
    component_loss      -> Eq. 15 / 28   (expected hinge loss, one Gaussian)
    component_grad_w    -> per-component part of Eq. 18 / 31
    component_grad_b    -> per-component part of Eq. 19 / 32
    gmmu_objective      -> Eq. 30        (full objective)
    gmmu_gradients      -> Eqs. 31-32    (full gradients for a mini-batch)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

# ---------------------------------------------------------------------------
# Numerical safety constant.  When d_sigma falls below this value we treat
# the Gaussian as a point mass and fall back to the standard hinge loss.
# This avoids division by zero in d_mu / d_sigma.
# ---------------------------------------------------------------------------
_EPS = 1e-12


# ===================================================================
# Building blocks (per-component)
# ===================================================================


def compute_d_mu(
    w: NDArray[np.floating],
    b: float,
    mu: NDArray[np.floating],
    y: float,
) -> float:
    """Signed margin distance of a component mean.

    Equation 13 (GSU) / 25 (GMMU):
        d_mu = 1 - y * (w^T mu + b)

    This is exactly the standard hinge-loss argument evaluated at the
    component mean *mu*.  It is positive when the mean is inside the
    margin or misclassified, and negative when it is correctly classified
    and outside the margin.

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    mu : ndarray of shape (d,)
        Mean of one Gaussian component.
    y : {+1, -1}
        Class label.

    Returns
    -------
    float
        The scalar d_mu.
    """
    return 1.0 - y * (w @ mu + b)


def compute_d_sigma(
    w: NDArray[np.floating],
    cov: NDArray[np.floating],
) -> float:
    """Uncertainty spread in the classification-relevant direction.

    Equation 14 (GSU) / 26 (GMMU):
        d_sigma = sqrt(2 * w^T Sigma w)

    The quantity w^T Sigma w is the variance of the projection w^T x
    when x ~ N(mu, Sigma).  So d_sigma captures how much the Gaussian
    is spread out in the direction perpendicular to the decision
    boundary (since w is the boundary normal).

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    cov : ndarray of shape (d,) or (d, d)
        Covariance of one Gaussian component.
        - shape (d,):   diagonal covariance (vector of variances).
        - shape (d, d): full covariance matrix.

    Returns
    -------
    float
        The non-negative scalar d_sigma.
    """
    if cov.ndim == 1:
        # Diagonal covariance: w^T Sigma w = sum(w_j^2 * sigma_jj)
        quad = w @ (cov * w)
    else:
        # Full covariance: w^T Sigma w
        quad = w @ cov @ w

    # Guard against negative values from floating-point noise.
    return np.sqrt(2.0 * max(quad, 0.0))


def component_loss(d_mu: float, d_sigma: float) -> float:
    """Closed-form expected hinge loss for a single Gaussian component.

    Equation 15 (GSU) / 28 (GMMU):
        L = (d_mu / 2) * [erf(d_mu / d_sigma) + 1]
            + (d_sigma / (2 sqrt(pi))) * exp(-(d_mu / d_sigma)^2)

    When d_sigma -> 0 (no uncertainty), this smoothly reduces to the
    standard hinge loss max(0, d_mu), as shown in Section 3.6 of the
    paper.

    Parameters
    ----------
    d_mu : float
        Output of ``compute_d_mu``.
    d_sigma : float
        Output of ``compute_d_sigma``.  Must be >= 0.

    Returns
    -------
    float
        The non-negative expected hinge loss.
    """
    # -- Limiting case: zero uncertainty -> standard hinge loss ----------
    if d_sigma < _EPS:
        return max(0.0, d_mu)

    ratio = d_mu / d_sigma
    erf_term = erf(ratio)
    exp_term = np.exp(-(ratio**2))

    loss = 0.5 * d_mu * (erf_term + 1.0) + (d_sigma / (2.0 * np.sqrt(np.pi))) * exp_term
    return float(loss)


def component_grad_w(
    w: NDArray[np.floating],
    mu: NDArray[np.floating],
    y: float,
    cov: NDArray[np.floating],
    d_mu: float,
    d_sigma: float,
) -> NDArray[np.floating]:
    """Gradient of one component's loss with respect to w.

    This is the per-component piece inside the sum of Equation 18 (GSU) /
    31 (GMMU), without the mixing weight or the 1/n factor:

        dL_i^(m)/dw = [exp(-r^2) / (sqrt(pi) * d_sigma)] * Sigma w
                      - (y / 2) * [erf(r) + 1] * mu

    where r = d_mu / d_sigma.

    The first term comes from the dependence of d_sigma on w through the
    quadratic form w^T Sigma w.  It has no analog in the standard SVM and
    is the new contribution due to uncertainty.

    The second term is analogous to the standard SVM hinge-loss gradient
    but uses a soft sigmoid-like factor (erf(r) + 1) / 2 instead of a
    hard step function.

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    mu : ndarray of shape (d,)
        Component mean.
    y : {+1, -1}
        Class label.
    cov : ndarray of shape (d,) or (d, d)
        Component covariance (diagonal or full).
    d_mu : float
        Pre-computed ``compute_d_mu`` value.
    d_sigma : float
        Pre-computed ``compute_d_sigma`` value.

    Returns
    -------
    ndarray of shape (d,)
        Gradient contribution from this component.
    """
    d = w.shape[0]

    # -- Limiting case: zero uncertainty -> subgradient of hinge loss ----
    if d_sigma < _EPS:
        if d_mu > 0.0:
            return -y * mu
        else:
            return np.zeros(d, dtype=w.dtype)

    ratio = d_mu / d_sigma
    erf_term = erf(ratio)
    exp_term = np.exp(-(ratio**2))

    # Sigma @ w (works for both diagonal and full covariance)
    if cov.ndim == 1:
        sigma_w = cov * w
    else:
        sigma_w = cov @ w

    # Two terms of the gradient
    uncertainty_term = (exp_term / (np.sqrt(np.pi) * d_sigma)) * sigma_w
    mean_term = (y / 2.0) * (erf_term + 1.0) * mu

    return uncertainty_term - mean_term


def component_grad_b(y: float, d_mu: float, d_sigma: float) -> float:
    """Gradient of one component's loss with respect to b.

    This is the per-component piece inside the sum of Equation 19 (GSU) /
    32 (GMMU), without the mixing weight or the 1/n factor:

        dL_i^(m)/db = -y * [erf(d_mu / d_sigma) + 1]

    Since d_sigma does not depend on b (b does not appear in w^T Sigma w),
    only the d_mu pathway contributes.  The factor (erf(r) + 1) / 2 acts
    as a soft version of the indicator function that the standard SVM uses.

    Parameters
    ----------
    y : {+1, -1}
        Class label.
    d_mu : float
        Pre-computed ``compute_d_mu`` value.
    d_sigma : float
        Pre-computed ``compute_d_sigma`` value.

    Returns
    -------
    float
        Gradient contribution from this component.
    """
    # -- Limiting case: zero uncertainty -> subgradient of hinge loss ----
    if d_sigma < _EPS:
        if d_mu > 0.0:
            return -y  # matches -y * (erf(+inf) + 1) / 2 = -y * 1
        else:
            return 0.0

    ratio = d_mu / d_sigma
    return -y * (erf(ratio) + 1.0) / 2.0


# ===================================================================
# Full objective and gradients (aggregate over samples and components)
# ===================================================================


def gmmu_objective(
    w: NDArray[np.floating],
    b: float,
    sample_uncertainty: list[dict],
    y: NDArray[np.floating],
    lam: float,
) -> float:
    """Full SVM-GMMU objective function.

    Equation 30:
        J(w, b) = (lam / 2) ||w||^2
                  + (1/n) sum_i sum_m pi_i^(m) L_i^(m)(w, b)

    where L_i^(m) is the closed-form expected hinge loss for the m-th
    Gaussian component of the i-th sample (Equation 28).

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    sample_uncertainty : list of dict
        Each dict has keys "weights", "means", "covariances".
    y : ndarray of shape (n,)
        Class labels in {+1, -1}.
    lam : float
        Regularization parameter (lambda in the paper).

    Returns
    -------
    float
        The objective value.
    """
    n = len(sample_uncertainty)
    reg = 0.5 * lam * (w @ w)

    total_loss = 0.0
    for i in range(n):
        su = sample_uncertainty[i]
        weights = su["weights"]
        means = su["means"]
        covs = su["covariances"]

        for m in range(len(weights)):
            d_mu = compute_d_mu(w, b, means[m], y[i])
            d_sig = compute_d_sigma(w, covs[m])
            total_loss += weights[m] * component_loss(d_mu, d_sig)

    return reg + total_loss / n


def gmmu_gradients(
    w: NDArray[np.floating],
    b: float,
    sample_uncertainty: list[dict],
    y: NDArray[np.floating],
    lam: float,
    batch_indices: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.floating], float]:
    """Gradients of the SVM-GMMU objective for a mini-batch.

    Equations 31-32:
        dJ/dw = lam * w
                + (1/|B|) sum_{i in B} sum_m pi_i^(m) dL_i^(m)/dw

        dJ/db = (1/|B|) sum_{i in B} sum_m pi_i^(m) dL_i^(m)/db

    When batch_indices is None, the gradient is computed over all samples
    (full-batch mode).

    Parameters
    ----------
    w : ndarray of shape (d,)
        Weight vector.
    b : float
        Bias term.
    sample_uncertainty : list of dict
        Each dict has keys "weights", "means", "covariances".
    y : ndarray of shape (n,)
        Class labels in {+1, -1}.
    lam : float
        Regularization parameter.
    batch_indices : ndarray of int or None
        Indices of the mini-batch samples.  None means use all samples.

    Returns
    -------
    grad_w : ndarray of shape (d,)
        Gradient with respect to w.
    grad_b : float
        Gradient with respect to b.
    """
    if batch_indices is None:
        batch_indices = np.arange(len(sample_uncertainty))

    batch_size = len(batch_indices)
    d = w.shape[0]

    grad_w = lam * w.copy()
    grad_b = 0.0

    for i in batch_indices:
        su = sample_uncertainty[i]
        weights = su["weights"]
        means = su["means"]
        covs = su["covariances"]

        for m in range(len(weights)):
            d_mu = compute_d_mu(w, b, means[m], y[i])
            d_sig = compute_d_sigma(w, covs[m])

            gw = component_grad_w(w, means[m], y[i], covs[m], d_mu, d_sig)
            gb = component_grad_b(y[i], d_mu, d_sig)

            grad_w += (weights[m] / batch_size) * gw
            grad_b += (weights[m] / batch_size) * gb

    return grad_w, grad_b
