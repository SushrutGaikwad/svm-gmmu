"""Tests for the closed-form loss and gradient functions in _loss.py.

Test strategy
-------------
1. **Known-value tests**: check d_mu, d_sigma against hand-computed values.
2. **Limiting behavior**: when covariance -> 0, the expected hinge loss
   must equal the standard hinge loss max(0, d_mu).  (Section 3.6)
3. **Non-negativity**: the expected hinge loss is always >= 0.
4. **Gradient correctness**: compare analytic gradients against centered
   finite differences.  This is the most important test -- if the math
   is wrong, the gradients will disagree with the numerical approximation.
5. **GMM vs GSU**: a GMM with one component must give the same loss and
   gradients as the single-Gaussian formulas.
"""

import numpy as np
import pytest
from scipy.special import erf

from svm_gmmu._loss import (
    component_grad_b,
    component_grad_w,
    component_loss,
    compute_d_mu,
    compute_d_sigma,
    gmmu_gradients,
    gmmu_objective,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def simple_2d():
    """A simple 2D setup for quick sanity checks."""
    w = np.array([1.0, 0.0])
    b = 0.0
    mu = np.array([2.0, 3.0])
    y = 1.0
    cov_diag = np.array([0.5, 0.5])
    cov_full = np.diag([0.5, 0.5])
    return w, b, mu, y, cov_diag, cov_full


# ===================================================================
# Tests for compute_d_mu
# ===================================================================


class TestComputeDMu:
    def test_basic(self, simple_2d):
        w, b, mu, y, _, _ = simple_2d
        # d_mu = 1 - y * (w^T mu + b) = 1 - 1 * (2 + 0) = -1
        assert compute_d_mu(w, b, mu, y) == pytest.approx(-1.0)

    def test_negative_label(self, simple_2d):
        w, b, mu, _, _, _ = simple_2d
        y = -1.0
        # d_mu = 1 - (-1) * (2 + 0) = 1 + 2 = 3
        assert compute_d_mu(w, b, mu, y) == pytest.approx(3.0)

    def test_with_bias(self):
        w = np.array([1.0, 1.0])
        b = -0.5
        mu = np.array([1.0, 0.5])
        y = 1.0
        # w^T mu + b = 1 + 0.5 - 0.5 = 1.0;  d_mu = 1 - 1 = 0
        assert compute_d_mu(w, b, mu, y) == pytest.approx(0.0)


# ===================================================================
# Tests for compute_d_sigma
# ===================================================================


class TestComputeDSigma:
    def test_diagonal(self, simple_2d):
        w, _, _, _, cov_diag, _ = simple_2d
        # w = [1, 0], cov = [0.5, 0.5]
        # w^T Sigma w = 1^2 * 0.5 + 0^2 * 0.5 = 0.5
        # d_sigma = sqrt(2 * 0.5) = 1.0
        assert compute_d_sigma(w, cov_diag) == pytest.approx(1.0)

    def test_full_matches_diagonal(self, simple_2d):
        w, _, _, _, cov_diag, cov_full = simple_2d
        d_diag = compute_d_sigma(w, cov_diag)
        d_full = compute_d_sigma(w, cov_full)
        assert d_diag == pytest.approx(d_full)

    def test_zero_covariance(self):
        w = np.array([1.0, 2.0])
        cov = np.array([0.0, 0.0])
        assert compute_d_sigma(w, cov) == pytest.approx(0.0)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            d = rng.integers(2, 10)
            w = rng.standard_normal(d)
            cov = np.abs(rng.standard_normal(d))  # diagonal, non-negative
            assert compute_d_sigma(w, cov) >= 0.0


# ===================================================================
# Tests for component_loss
# ===================================================================


class TestComponentLoss:
    def test_zero_uncertainty_correct_side(self):
        """When sigma=0 and mean is correctly classified (d_mu < 0),
        loss should be 0 (standard hinge loss)."""
        assert component_loss(-2.0, 0.0) == pytest.approx(0.0)

    def test_zero_uncertainty_wrong_side(self):
        """When sigma=0 and mean is misclassified (d_mu > 0),
        loss should be d_mu (standard hinge loss)."""
        assert component_loss(1.5, 0.0) == pytest.approx(1.5)

    def test_zero_uncertainty_on_margin(self):
        """When sigma=0 and d_mu=0, loss should be 0."""
        assert component_loss(0.0, 0.0) == pytest.approx(0.0)

    def test_non_negative(self):
        """Expected hinge loss is always >= 0."""
        rng = np.random.default_rng(123)
        for _ in range(200):
            d_mu = rng.uniform(-5.0, 5.0)
            d_sigma = rng.uniform(0.0, 5.0)
            loss = component_loss(d_mu, d_sigma)
            assert loss >= -1e-15, f"Negative loss: {loss}"

    def test_large_uncertainty_increases_loss(self):
        """For a correctly classified mean (d_mu < 0), increasing
        uncertainty should increase the loss because more probability
        mass crosses the boundary."""
        d_mu = -1.0
        loss_small = component_loss(d_mu, 0.1)
        loss_large = component_loss(d_mu, 2.0)
        assert loss_large > loss_small

    def test_known_value(self):
        """Check against a hand-computed value.
        d_mu = 1.0, d_sigma = 1.0
        ratio = 1.0
        erf(1.0) ~ 0.8427
        exp(-1.0) ~ 0.3679
        loss = 0.5 * 1.0 * (0.8427 + 1) + (1.0 / (2*sqrt(pi))) * 0.3679
             = 0.5 * 1.8427 + 0.2821 * 0.3679
             = 0.9213 + 0.1038
             ~ 1.0251
        """
        loss = component_loss(1.0, 1.0)
        assert loss == pytest.approx(1.0251, abs=0.001)


# ===================================================================
# Tests for gradients via finite differences
# ===================================================================


class TestGradients:
    """Compare analytic gradients against centered finite differences.

    This is the most critical test.  If the closed-form gradient
    formulas (Eqs. 31-32) are wrong, the finite-difference
    approximation will disagree.
    """

    @staticmethod
    def _numerical_grad_w(w, b, mu, y, cov, eps=1e-5):
        """Centered finite-difference gradient w.r.t. w."""
        d = w.shape[0]
        grad = np.zeros(d)
        for j in range(d):
            w_plus = w.copy()
            w_plus[j] += eps
            w_minus = w.copy()
            w_minus[j] -= eps
            d_mu_p = compute_d_mu(w_plus, b, mu, y)
            d_sig_p = compute_d_sigma(w_plus, cov)
            loss_p = component_loss(d_mu_p, d_sig_p)

            d_mu_m = compute_d_mu(w_minus, b, mu, y)
            d_sig_m = compute_d_sigma(w_minus, cov)
            loss_m = component_loss(d_mu_m, d_sig_m)

            grad[j] = (loss_p - loss_m) / (2 * eps)
        return grad

    @staticmethod
    def _numerical_grad_b(w, b, mu, y, cov, eps=1e-5):
        """Centered finite-difference gradient w.r.t. b."""
        d_mu_p = compute_d_mu(w, b + eps, mu, y)
        d_sig = compute_d_sigma(w, cov)  # does not depend on b
        loss_p = component_loss(d_mu_p, d_sig)

        d_mu_m = compute_d_mu(w, b - eps, mu, y)
        loss_m = component_loss(d_mu_m, d_sig)

        return (loss_p - loss_m) / (2 * eps)

    def test_grad_w_diagonal(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            d = rng.integers(2, 8)
            w = rng.standard_normal(d)
            b = rng.standard_normal()
            mu = rng.standard_normal(d)
            y = rng.choice([-1.0, 1.0])
            cov = np.abs(rng.standard_normal(d)) + 0.01

            d_mu = compute_d_mu(w, b, mu, y)
            d_sig = compute_d_sigma(w, cov)
            analytic = component_grad_w(w, mu, y, cov, d_mu, d_sig)
            numerical = self._numerical_grad_w(w, b, mu, y, cov)
            np.testing.assert_allclose(analytic, numerical, atol=1e-4)

    def test_grad_w_full(self):
        rng = np.random.default_rng(8)
        for _ in range(20):
            d = rng.integers(2, 6)
            w = rng.standard_normal(d)
            b = rng.standard_normal()
            mu = rng.standard_normal(d)
            y = rng.choice([-1.0, 1.0])
            # Create a random positive definite matrix
            A = rng.standard_normal((d, d))
            cov = A @ A.T + 0.01 * np.eye(d)

            d_mu = compute_d_mu(w, b, mu, y)
            d_sig = compute_d_sigma(w, cov)
            analytic = component_grad_w(w, mu, y, cov, d_mu, d_sig)
            numerical = self._numerical_grad_w(w, b, mu, y, cov)
            np.testing.assert_allclose(analytic, numerical, atol=1e-4)

    def test_grad_b(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            d = rng.integers(2, 8)
            w = rng.standard_normal(d)
            b = rng.standard_normal()
            mu = rng.standard_normal(d)
            y = rng.choice([-1.0, 1.0])
            cov = np.abs(rng.standard_normal(d)) + 0.01

            d_mu = compute_d_mu(w, b, mu, y)
            d_sig = compute_d_sigma(w, cov)
            analytic = component_grad_b(y, d_mu, d_sig)
            numerical = self._numerical_grad_b(w, b, mu, y, cov)
            assert analytic == pytest.approx(numerical, abs=1e-4)


# ===================================================================
# Tests for full objective and gradients (GMM level)
# ===================================================================


class TestGmmuObjective:
    def test_single_component_matches_gsu(self):
        """A GMM with M=1 must give the same loss as the direct
        component_loss call."""
        rng = np.random.default_rng(10)
        d = 3
        w = rng.standard_normal(d)
        b = rng.standard_normal()
        mu = rng.standard_normal(d)
        y_val = 1.0
        cov = np.abs(rng.standard_normal(d)) + 0.01
        lam = 0.1

        su = [
            {
                "weights": np.array([1.0]),
                "means": mu.reshape(1, -1),
                "covariances": cov.reshape(1, -1),
            }
        ]
        y = np.array([y_val])

        obj = gmmu_objective(w, b, su, y, lam)

        # Manual computation
        d_mu = compute_d_mu(w, b, mu, y_val)
        d_sig = compute_d_sigma(w, cov)
        expected = 0.5 * lam * (w @ w) + component_loss(d_mu, d_sig)
        assert obj == pytest.approx(expected)

    def test_objective_decreases_with_optimization(self):
        """A few gradient steps should decrease the objective."""
        rng = np.random.default_rng(11)
        d = 2
        w = rng.standard_normal(d)
        b = 0.0
        lam = 0.01

        su = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0, 0.0]]),
                "covariances": np.array([[0.1, 0.1]]),
            },
            {
                "weights": np.array([1.0]),
                "means": np.array([[-1.0, 0.0]]),
                "covariances": np.array([[0.1, 0.1]]),
            },
        ]
        y = np.array([1.0, -1.0])

        obj_before = gmmu_objective(w, b, su, y, lam)

        # Take 50 gradient steps
        for t in range(1, 51):
            eta = 1.0 / (lam * t)
            gw, gb = gmmu_gradients(w, b, su, y, lam)
            w = w - eta * gw
            b = b - eta * gb

        obj_after = gmmu_objective(w, b, su, y, lam)
        assert obj_after < obj_before

    def test_full_objective_gradient_finite_diff(self):
        """Check the full gmmu_gradients against finite differences
        on the full gmmu_objective."""
        rng = np.random.default_rng(12)
        d = 3
        w = rng.standard_normal(d)
        b = rng.standard_normal()
        lam = 0.05

        su = [
            {
                "weights": np.array([0.6, 0.4]),
                "means": rng.standard_normal((2, d)),
                "covariances": np.abs(rng.standard_normal((2, d))) + 0.01,
            },
            {
                "weights": np.array([1.0]),
                "means": rng.standard_normal((1, d)),
                "covariances": np.abs(rng.standard_normal((1, d))) + 0.01,
            },
        ]
        y = np.array([1.0, -1.0])

        # Analytic
        grad_w, grad_b = gmmu_gradients(w, b, su, y, lam)

        # Numerical grad_w
        eps = 1e-5
        num_gw = np.zeros(d)
        for j in range(d):
            w_p = w.copy()
            w_p[j] += eps
            w_m = w.copy()
            w_m[j] -= eps
            num_gw[j] = (
                gmmu_objective(w_p, b, su, y, lam) - gmmu_objective(w_m, b, su, y, lam)
            ) / (2 * eps)

        np.testing.assert_allclose(grad_w, num_gw, atol=1e-4)

        # Numerical grad_b
        num_gb = (
            gmmu_objective(w, b + eps, su, y, lam)
            - gmmu_objective(w, b - eps, su, y, lam)
        ) / (2 * eps)
        assert grad_b == pytest.approx(num_gb, abs=1e-4)
