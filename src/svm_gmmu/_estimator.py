"""Scikit-learn compatible SVM-GMMU estimator.

This module provides :class:`SvmGmmu`, a linear classifier that accounts
for per-sample uncertainty modeled as Gaussian mixtures.  It follows the
scikit-learn estimator API (``fit`` / ``predict`` / ``decision_function``)
and uses the Pegasos-style SGD algorithm described in Section 3.7 of the
paper.

Special cases
-------------
- When ``sample_uncertainty`` has one component per sample, the model is
  equivalent to SVM-GSU (Part III of the paper).
- When ``sample_uncertainty`` is ``None`` (or all covariances are zero),
  the model reduces to a standard linear SVM (Section 3.6).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from svm_gmmu._loss import gmmu_gradients, gmmu_objective
from svm_gmmu._validation import (
    build_default_uncertainty,
    validate_labels,
    validate_sample_uncertainty,
)


class SvmGmmu(BaseEstimator, ClassifierMixin):
    """SVM with Gaussian Mixture Model Uncertainty.

    A linear classifier that minimizes the expected hinge loss under
    per-sample Gaussian mixture uncertainty, solved via Pegasos-style
    stochastic gradient descent.

    Parameters
    ----------
    lam : float, default=1e-2
        Regularization parameter (called lambda in the paper).  Larger
        values produce a wider margin at the cost of more training
        errors.  Must be positive.

    max_iter : int, default=1000
        Number of SGD iterations (called *T* in the paper).

    batch_size : int, default=32
        Number of samples per mini-batch (called *k* in the paper).
        Set to 1 for pure stochastic gradient descent.

    random_state : int or None, default=None
        Seed for the random number generator used in mini-batch
        sampling.  Pass an integer for reproducible results.

    verbose : bool, default=False
        If True, print the objective value every ``log_interval``
        iterations.

    log_interval : int, default=100
        How often to print the objective when ``verbose=True``.

    Attributes
    ----------
    coef_ : ndarray of shape (d,)
        Learned weight vector after fitting.

    intercept_ : float
        Learned bias term after fitting.

    n_features_in_ : int
        Number of features seen during ``fit``.

    classes_ : ndarray of shape (2,)
        The class labels, always ``array([-1, 1])``.

    loss_history_ : list of float
        Objective value recorded every ``log_interval`` iterations
        (only populated when ``verbose=True``).

    Examples
    --------
    >>> import numpy as np
    >>> from svm_gmmu import SvmGmmu
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> y = np.array([1, -1])
    >>> model = SvmGmmu(lam=0.01, max_iter=500)
    >>> model.fit(X, y)  # no uncertainty -> standard SVM
    SvmGmmu(lam=0.01, max_iter=500)
    >>> model.predict(X)
    array([ 1., -1.])
    """

    def __init__(
        self,
        lam: float = 1e-2,
        max_iter: int = 1000,
        batch_size: int = 32,
        random_state: int | None = None,
        verbose: bool = False,
        log_interval: int = 100,
    ) -> None:
        # Scikit-learn convention: store every __init__ parameter as-is.
        # No validation here; that happens in fit().
        self.lam = lam
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.log_interval = log_interval

    # ------------------------------------------------------------------ #
    #  fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        sample_uncertainty: list[dict] | None = None,
    ) -> "SvmGmmu":
        """Fit the SVM-GMMU model using Pegasos-style SGD.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.  When ``sample_uncertainty`` is
            provided, each row should be the overall mixture mean
            (weighted average of component means).  When
            ``sample_uncertainty`` is None, rows are the fixed data
            points (standard SVM mode).

        y : ndarray of shape (n_samples,)
            Class labels, must be in {+1, -1}.

        sample_uncertainty : list of dict or None, default=None
            Per-sample GMM uncertainty.  Each dict must have keys:

            - ``"weights"``: array of shape (M_i,), non-negative,
              summing to 1.
            - ``"means"``: array of shape (M_i, n_features).
            - ``"covariances"``: array of shape (M_i, n_features) for
              diagonal covariances, or (M_i, n_features, n_features)
              for full covariance matrices.

            If None, the model trains as a standard linear SVM with
            zero uncertainty.

        Returns
        -------
        self
            The fitted estimator.
        """
        # -- Validate hyperparameters -----------------------------------
        if self.lam <= 0:
            raise ValueError(f"lam must be positive, got {self.lam}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")

        # -- Validate X and y -------------------------------------------
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        n, d = X.shape

        y = validate_labels(y)
        if y.shape[0] != n:
            raise ValueError(f"X has {n} samples but y has {y.shape[0]} labels.")

        # -- Validate or build uncertainty ------------------------------
        if sample_uncertainty is None:
            sample_uncertainty = build_default_uncertainty(X)
        else:
            sample_uncertainty = validate_sample_uncertainty(sample_uncertainty, n, d)

        # -- Store metadata ---------------------------------------------
        self.n_features_in_ = d
        self.classes_ = np.array([-1.0, 1.0])
        self.loss_history_: list[float] = []

        # -- Run Pegasos-style SGD (Section 3.7 of the paper) -----------
        self.coef_, self.intercept_ = self._pegasos_sgd(X, y, sample_uncertainty)

        return self

    # ------------------------------------------------------------------ #
    #  predict / decision_function
    # ------------------------------------------------------------------ #

    def decision_function(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Compute the signed distance to the decision boundary.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for prediction.  No uncertainty is used at
            test time.

        Returns
        -------
        ndarray of shape (n_samples,)
            Values of w^T x + b for each sample.
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Predict class labels for the given samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for prediction.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels in {+1, -1}.
        """
        scores = self.decision_function(X)
        return np.sign(scores)

    # ------------------------------------------------------------------ #
    #  Pegasos SGD (private)
    # ------------------------------------------------------------------ #

    def _pegasos_sgd(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_uncertainty: list[dict],
    ) -> tuple[NDArray[np.float64], float]:
        """Run the Pegasos-style SGD solver.

        This implements the algorithm from Section 3.7 of the paper:

        1. Initialize w with ||w|| <= 1/sqrt(lam), b = 0.
        2. For t = 1, ..., T:
           a. Sample a mini-batch of indices.
           b. Set learning rate eta_t = 1 / (lam * t).
           c. Compute gradients using Eqs. 31-32.
           d. Update w and b.
           e. Project w so that ||w|| <= 1/sqrt(lam).

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Feature matrix (used only for computing the objective when
            verbose=True).
        y : ndarray of shape (n,)
            Class labels.
        sample_uncertainty : list of dict
            Validated uncertainty data.

        Returns
        -------
        w : ndarray of shape (d,)
            Learned weight vector.
        b : float
            Learned bias.
        """
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        # -- Initialization (Step 2 of the algorithm) -------------------
        # w is initialized with ||w|| <= 1/sqrt(lam).
        # A simple choice: small random values scaled appropriately.
        w_bound = 1.0 / np.sqrt(self.lam)
        w = rng.standard_normal(d).astype(np.float64)
        w *= w_bound / (np.linalg.norm(w) * d)  # small initial norm
        b = 0.0

        effective_batch = min(self.batch_size, n)

        # -- SGD loop (Step 3 of the algorithm) -------------------------
        for t in range(1, self.max_iter + 1):
            # (a) Sample mini-batch
            batch_idx = rng.choice(n, size=effective_batch, replace=False)

            # (b) Learning rate: eta_t = 1 / (lam * t)
            eta = 1.0 / (self.lam * t)

            # (c) Compute gradients (Eqs. 31-32)
            grad_w, grad_b = gmmu_gradients(
                w, b, sample_uncertainty, y, self.lam, batch_idx
            )

            # (d) Update w and b
            w = w - eta * grad_w
            b = b - eta * grad_b

            # (e) Project: ||w|| <= 1/sqrt(lam)
            w_norm = np.linalg.norm(w)
            if w_norm > w_bound:
                w *= w_bound / w_norm

            # -- Logging ------------------------------------------------
            if self.verbose and t % self.log_interval == 0:
                obj = gmmu_objective(w, b, sample_uncertainty, y, self.lam)
                self.loss_history_.append(obj)
                print(f"  iter {t:>6d} / {self.max_iter}  |  objective = {obj:.6f}")

        return w, b
