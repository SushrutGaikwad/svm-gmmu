"""Visualization utilities for SVM-GMMU.

This module provides three plotting functions for 2-D datasets:

    plot_uncertainty
        Show per-sample GMM density contours at configurable sigma levels.
    plot_boundary
        Show the decision boundary and margins learned by a fitted
        :class:`~svm_gmmu.SvmGmmu` model, overlaid on the GMM contours.
    plot_boundary_comparison
        Side-by-side comparison of two fitted models (typically SVM-GMMU
        vs. a standard SVM) on the same dataset.

All three functions require ``matplotlib`` (an optional dependency).  Install
it with ``uv sync --extra dev`` or ``pip install matplotlib``.

.. note::

   These functions only support 2-D feature spaces (``d = 2``).  Higher-
   dimensional data must be projected to 2-D before plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import chi2, multivariate_normal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from svm_gmmu._estimator import SvmGmmu


# ===================================================================
# Module-level defaults
# ===================================================================

_CLASS_COLORS: dict[int, str] = {+1: "#2563eb", -1: "#dc2626"}
_CLASS_MARKERS: dict[int, str] = {+1: "o", -1: "s"}
_CLASS_LABELS: dict[int, str] = {+1: "+1", -1: "\u22121"}

_DEFAULT_SIGMAS: tuple[int, ...] = (1, 2, 3)
_DEFAULT_GRID_RES: int = 300
_DEFAULT_MC_SAMPLES: int = 200_000


# ===================================================================
# Private helpers
# ===================================================================


def _require_matplotlib():
    """Import and return ``matplotlib.pyplot``, raising a clear error if
    matplotlib is not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting.  Install it with "
            "'uv sync --extra dev' or 'pip install matplotlib'."
        ) from exc


def _check_2d(X: NDArray, sample_uncertainty: list[dict]) -> None:
    """Raise ``ValueError`` if the data is not 2-D."""
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Plotting requires d=2 features, got X with shape {X.shape}.")
    for i, su in enumerate(sample_uncertainty):
        if su["means"].shape[1] != 2:
            raise ValueError(
                f"sample_uncertainty[{i}]['means'] has {su['means'].shape[1]} "
                f"features, expected 2."
            )


def _gmm_pdf(
    x1: NDArray,
    x2: NDArray,
    su: dict,
) -> NDArray:
    """Evaluate a GMM density on a 2-D meshgrid.

    Parameters
    ----------
    x1, x2 : ndarray
        2-D arrays from ``np.meshgrid``.
    su : dict
        Single sample's uncertainty dict with keys ``"weights"``,
        ``"means"``, ``"covariances"``.

    Returns
    -------
    ndarray
        Density values, same shape as *x1*.
    """
    pos = np.stack([x1, x2], axis=-1)
    Z = np.zeros_like(x1)
    for m in range(len(su["weights"])):
        rv = multivariate_normal(mean=su["means"][m], cov=su["covariances"][m])
        Z += su["weights"][m] * rv.pdf(pos)
    return Z


def _sigma_thresholds(
    su: dict,
    sigmas: tuple[int, ...],
    n_mc: int = _DEFAULT_MC_SAMPLES,
    rng: np.random.Generator | None = None,
) -> dict[int, float]:
    """Find GMM density values that enclose given sigma levels of mass.

    For a single 2-D Gaussian the *k*-sigma contour encloses probability
    ``chi2.cdf(k**2, df=2)``.  For a GMM there is no closed form, so we
    estimate the density threshold by Monte Carlo sampling.

    Parameters
    ----------
    su : dict
        One sample's GMM parameters.
    sigmas : tuple of int
        Sigma levels (e.g. ``(1, 2, 3)``).
    n_mc : int
        Number of Monte Carlo samples for threshold estimation.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    dict mapping each sigma value to its density threshold.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = su["weights"]
    means = su["means"]
    covs = su["covariances"]
    n_components = len(weights)

    probs = {k: chi2.cdf(k**2, df=2) for k in sigmas}

    # Draw from the GMM.
    comp_idx = rng.choice(n_components, size=n_mc, p=weights)
    samples = np.empty((n_mc, 2))
    for m in range(n_components):
        mask = comp_idx == m
        count = mask.sum()
        if count > 0:
            samples[mask] = rng.multivariate_normal(means[m], covs[m], size=count)

    # Evaluate the full GMM density at each sample.
    densities = np.zeros(n_mc)
    for m in range(n_components):
        rv = multivariate_normal(mean=means[m], cov=covs[m])
        densities += weights[m] * rv.pdf(samples)

    # The k-sigma contour is the density level above which the
    # enclosed probability equals ``probs[k]``.
    return {
        k: float(np.percentile(densities, 100.0 * (1.0 - probs[k]))) for k in sigmas
    }


def _draw_contours(
    ax: Axes,
    su: dict,
    color: str,
    sigmas: tuple[int, ...],
    grid_resolution: int,
    rng: np.random.Generator,
) -> None:
    """Draw filled + line contours for one sample's GMM on *ax*."""
    sorted_sigmas = sorted(sigmas)
    means_i = su["means"]
    pad = max(sorted_sigmas) * 0.8

    x1_grid = np.linspace(
        means_i[:, 0].min() - pad, means_i[:, 0].max() + pad, grid_resolution
    )
    x2_grid = np.linspace(
        means_i[:, 1].min() - pad, means_i[:, 1].max() + pad, grid_resolution
    )
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = _gmm_pdf(X1, X2, su)

    levels_dict = _sigma_thresholds(su, tuple(sorted_sigmas), rng=rng)

    # Density thresholds ordered from lowest (outermost contour) to
    # highest (innermost), which is what matplotlib expects.
    sigmas_rev = sorted_sigmas[::-1]
    lvl = [levels_dict[k] for k in sigmas_rev]

    # Filled bands.
    for k_idx in range(len(sigmas_rev)):
        upper = Z.max() * 10 if k_idx == len(sigmas_rev) - 1 else lvl[k_idx + 1]
        ax.contourf(
            X1,
            X2,
            Z,
            levels=[lvl[k_idx], upper],
            colors=[color],
            alpha=0.04 + 0.05 * k_idx,
        )

    # Contour lines with sigma labels.
    cs = ax.contour(
        X1,
        X2,
        Z,
        levels=lvl,
        colors=[color],
        linewidths=0.8,
        alpha=0.6,
    )
    fmt = {lvl[j]: f"{sigmas_rev[j]}\u03c3" for j in range(len(sigmas_rev))}
    ax.clabel(cs, inline=True, fontsize=7, fmt=fmt)


def _draw_points(
    ax: Axes,
    X: NDArray,
    y: NDArray,
    point_size: float = 90,
) -> None:
    """Scatter the observed points on *ax* with class-dependent styling."""
    for cls in [+1, -1]:
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=point_size,
            c=_CLASS_COLORS[cls],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label=f"Class {_CLASS_LABELS[cls]}",
            marker=_CLASS_MARKERS[cls],
        )


def _draw_boundary(
    ax: Axes,
    model: SvmGmmu,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_resolution: int,
) -> None:
    """Draw the decision boundary (solid) and ±1 margins (dashed)."""
    x1_grid = np.linspace(xlim[0], xlim[1], grid_resolution)
    x2_grid = np.linspace(ylim[0], ylim[1], grid_resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])

    Z = model.decision_function(grid_pts).reshape(X1.shape)

    ax.contour(X1, X2, Z, levels=[0], colors=["k"], linewidths=2.0)
    ax.contour(
        X1,
        X2,
        Z,
        levels=[-1, 1],
        colors=["k"],
        linewidths=1.0,
        linestyles="dashed",
        alpha=0.5,
    )


def _add_legend(
    ax: Axes,
    sigmas: tuple[int, ...],
    include_boundary: bool = False,
) -> None:
    """Build a consolidated legend on *ax*."""
    from matplotlib.lines import Line2D

    handles, _ = ax.get_legend_handles_labels()

    if include_boundary:
        handles.append(Line2D([0], [0], color="k", lw=2, label="Decision boundary"))
        handles.append(
            Line2D(
                [0], [0], color="k", lw=1, ls="--", alpha=0.5, label="Margin (\u00b11)"
            )
        )

    sigma_str = ", ".join(f"{s}\u03c3" for s in sorted(sigmas))
    handles.append(
        Line2D([0], [0], color="gray", lw=0.8, label=f"Contours at {sigma_str}")
    )
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)


def _auto_limits(
    sample_uncertainty: list[dict],
    pad: float = 1.5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits that comfortably contain all GMM components."""
    all_means = np.vstack([su["means"] for su in sample_uncertainty])
    x1_lo = float(all_means[:, 0].min() - pad)
    x1_hi = float(all_means[:, 0].max() + pad)
    x2_lo = float(all_means[:, 1].min() - pad)
    x2_hi = float(all_means[:, 1].max() + pad)
    return (x1_lo, x1_hi), (x2_lo, x2_hi)


def _style_ax(ax: Axes, title: str | None, show_ylabel: bool = True) -> None:
    """Apply common axis styling."""
    if title is not None:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_1$", fontsize=12)
    if show_ylabel:
        ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# ===================================================================
# Public API
# ===================================================================


def plot_uncertainty(
    X: NDArray,
    y: NDArray,
    sample_uncertainty: list[dict],
    *,
    sigmas: tuple[int, ...] = _DEFAULT_SIGMAS,
    grid_resolution: int = _DEFAULT_GRID_RES,
    point_size: float = 100,
    figsize: tuple[float, float] = (10, 10),
    title: str = "GMM Uncertainty Contours",
    random_state: int | None = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot per-sample GMM density contours at configurable sigma levels.

    This is a data-only visualization — no model is needed.  Each sample's
    uncertainty is shown as nested filled contours (1σ, 2σ, …), colored
    by class label, with the observed points overlaid.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Observed feature matrix (2-D only).
    y : ndarray of shape (n,)
        Class labels in {+1, -1}.
    sample_uncertainty : list of dict
        Per-sample GMM uncertainty (see :meth:`SvmGmmu.fit`).
    sigmas : tuple of int, default=(1, 2, 3)
        Which sigma-level contours to draw.
    grid_resolution : int, default=300
        Grid points per axis for density evaluation.
    point_size : float, default=100
        Marker size for the observed points.
    figsize : tuple of float, default=(10, 10)
        Figure size in inches (ignored when *ax* is provided).
    title : str, default="GMM Uncertainty Contours"
        Axes title.
    random_state : int or None, default=0
        Seed for the Monte Carlo sigma-level estimation.  Pass ``None``
        for non-deterministic output.
    ax : matplotlib Axes or None, default=None
        If provided, draw on this axes instead of creating a new figure.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    ax : Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If the data is not 2-D.
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> from svm_gmmu import SvmGmmu
    >>> from svm_gmmu.plotting import plot_uncertainty
    >>> fig, ax = plot_uncertainty(X, y, sample_uncertainty)
    """
    plt = _require_matplotlib()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    _check_2d(X, sample_uncertainty)

    rng = np.random.default_rng(random_state)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i in range(len(X)):
        _draw_contours(
            ax,
            sample_uncertainty[i],
            _CLASS_COLORS[int(y[i])],
            sigmas,
            grid_resolution,
            rng,
        )

    _draw_points(ax, X, y, point_size)
    _add_legend(ax, sigmas, include_boundary=False)
    _style_ax(ax, title)

    return fig, ax


def plot_boundary(
    X: NDArray,
    y: NDArray,
    sample_uncertainty: list[dict],
    model: SvmGmmu,
    *,
    sigmas: tuple[int, ...] = _DEFAULT_SIGMAS,
    grid_resolution: int = _DEFAULT_GRID_RES,
    point_size: float = 90,
    figsize: tuple[float, float] = (10, 10),
    title: str = "SVM-GMMU Decision Boundary",
    random_state: int | None = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a fitted model's decision boundary over the GMM contours.

    Draws the decision boundary (``w^T x + b = 0``, solid black line)
    and the margin lines (``w^T x + b = ±1``, dashed) on top of the
    per-sample GMM density contours.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Observed feature matrix (2-D only).
    y : ndarray of shape (n,)
        Class labels in {+1, -1}.
    sample_uncertainty : list of dict
        Per-sample GMM uncertainty.
    model : SvmGmmu
        A fitted :class:`~svm_gmmu.SvmGmmu` instance.
    sigmas : tuple of int, default=(1, 2, 3)
        Which sigma-level contours to draw.
    grid_resolution : int, default=300
        Grid points per axis for density evaluation.
    point_size : float, default=90
        Marker size for the observed points.
    figsize : tuple of float, default=(10, 10)
        Figure size in inches (ignored when *ax* is provided).
    title : str, default="SVM-GMMU Decision Boundary"
        Axes title.
    random_state : int or None, default=0
        Seed for the Monte Carlo sigma-level estimation.
    ax : matplotlib Axes or None, default=None
        If provided, draw on this axes instead of creating a new figure.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    ax : Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If the data is not 2-D.
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> from svm_gmmu import SvmGmmu
    >>> from svm_gmmu.plotting import plot_boundary
    >>> model = SvmGmmu(lam=0.01, max_iter=5000, batch_size=1, random_state=42)
    >>> model.fit(X, y, sample_uncertainty=sample_uncertainty)
    SvmGmmu(batch_size=1, lam=0.01, max_iter=5000, random_state=42)
    >>> fig, ax = plot_boundary(X, y, sample_uncertainty, model)
    """
    plt = _require_matplotlib()
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(model, ["coef_", "intercept_"])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    _check_2d(X, sample_uncertainty)

    rng = np.random.default_rng(random_state)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i in range(len(X)):
        _draw_contours(
            ax,
            sample_uncertainty[i],
            _CLASS_COLORS[int(y[i])],
            sigmas,
            grid_resolution,
            rng,
        )

    xlim, ylim = _auto_limits(sample_uncertainty)
    _draw_boundary(ax, model, xlim, ylim, grid_resolution)
    _draw_points(ax, X, y, point_size)
    _add_legend(ax, sigmas, include_boundary=True)
    _style_ax(ax, title)

    return fig, ax


def plot_boundary_comparison(
    X: NDArray,
    y: NDArray,
    sample_uncertainty: list[dict],
    model_gmmu: SvmGmmu,
    model_svm: SvmGmmu,
    *,
    sigmas: tuple[int, ...] = _DEFAULT_SIGMAS,
    grid_resolution: int = _DEFAULT_GRID_RES,
    point_size: float = 90,
    figsize: tuple[float, float] = (16, 8),
    titles: tuple[str, str] = (
        "SVM-GMMU (with uncertainty)",
        "Standard SVM (no uncertainty)",
    ),
    suptitle: str = "Decision Boundary Comparison: SVM-GMMU vs Standard SVM",
    random_state: int | None = 0,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Side-by-side comparison of two fitted models on the same dataset.

    The left panel shows the *model_gmmu* boundary and the right panel
    shows the *model_svm* boundary, both overlaid on the same GMM
    density contours and observed points.  This is useful for seeing
    how uncertainty-aware training shifts the decision boundary compared
    to a standard (point-mass) SVM.

    Parameters
    ----------
    X : ndarray of shape (n, 2)
        Observed feature matrix (2-D only).
    y : ndarray of shape (n,)
        Class labels in {+1, -1}.
    sample_uncertainty : list of dict
        Per-sample GMM uncertainty.
    model_gmmu : SvmGmmu
        A fitted uncertainty-aware model.
    model_svm : SvmGmmu
        A fitted standard SVM model (trained without uncertainty).
    sigmas : tuple of int, default=(1, 2, 3)
        Which sigma-level contours to draw.
    grid_resolution : int, default=300
        Grid points per axis for density evaluation.
    point_size : float, default=90
        Marker size for the observed points.
    figsize : tuple of float, default=(16, 8)
        Figure size in inches.
    titles : tuple of str, default=("SVM-GMMU (with uncertainty)", ...)
        Titles for the left and right panels.
    suptitle : str
        Figure-level title.
    random_state : int or None, default=0
        Seed for the Monte Carlo sigma-level estimation.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    (ax_left, ax_right) : tuple of Axes
        The two subplot axes.

    Raises
    ------
    ValueError
        If the data is not 2-D.
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> from svm_gmmu import SvmGmmu
    >>> from svm_gmmu.plotting import plot_boundary_comparison
    >>> model_gmmu = SvmGmmu(lam=0.01, max_iter=5000, random_state=42)
    >>> model_gmmu.fit(X, y, sample_uncertainty=sample_uncertainty)
    SvmGmmu(lam=0.01, max_iter=5000, random_state=42)
    >>> model_svm = SvmGmmu(lam=0.01, max_iter=5000, random_state=42)
    >>> model_svm.fit(X, y)
    SvmGmmu(lam=0.01, max_iter=5000, random_state=42)
    >>> fig, (ax_l, ax_r) = plot_boundary_comparison(
    ...     X, y, sample_uncertainty, model_gmmu, model_svm,
    ... )
    """
    plt = _require_matplotlib()
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(model_gmmu, ["coef_", "intercept_"])
    check_is_fitted(model_svm, ["coef_", "intercept_"])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    _check_2d(X, sample_uncertainty)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    ax_left, ax_right = axes

    xlim, ylim = _auto_limits(sample_uncertainty)

    for ax, model, panel_title, is_left in [
        (ax_left, model_gmmu, titles[0], True),
        (ax_right, model_svm, titles[1], False),
    ]:
        # Use a fresh RNG seeded identically so both panels get the
        # same sigma-level thresholds for identical contour lines.
        rng = np.random.default_rng(random_state)

        for i in range(len(X)):
            _draw_contours(
                ax,
                sample_uncertainty[i],
                _CLASS_COLORS[int(y[i])],
                sigmas,
                grid_resolution,
                rng,
            )

        _draw_boundary(ax, model, xlim, ylim, grid_resolution)
        _draw_points(ax, X, y, point_size)
        _add_legend(ax, sigmas, include_boundary=True)
        _style_ax(ax, panel_title, show_ylabel=is_left)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    fig.subplots_adjust(wspace=0.08)

    return fig, (ax_left, ax_right)
