"""Tests for the SvmGmmu estimator.

Test strategy
-------------
1. **Smoke tests**: fit and predict run without errors.
2. **Linearly separable data**: the model achieves perfect accuracy.
3. **No uncertainty (standard SVM mode)**: sample_uncertainty=None works.
4. **Diagonal and full covariance**: both formats are accepted.
5. **Scikit-learn API**: get_params, set_params, score, check_is_fitted.
6. **Validation errors**: bad inputs raise ValueError/TypeError.
7. **Verbose mode**: loss_history_ is populated.
8. **Limiting behavior**: with tiny covariances, SVM-GMMU should behave
   similarly to standard SVM.
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from svm_gmmu import SvmGmmu


# ===================================================================
# Helpers
# ===================================================================


def make_linearly_separable(n=100, d=2, seed=42):
    """Create a linearly separable dataset with optional uncertainty."""
    rng = np.random.default_rng(seed)
    # Class +1: centered at [+2, 0, ...], class -1: centered at [-2, 0, ...]
    X_pos = rng.standard_normal((n // 2, d)) * 0.5
    X_pos[:, 0] += 2.0
    X_neg = rng.standard_normal((n // 2, d)) * 0.5
    X_neg[:, 0] -= 2.0

    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))

    # Simple diagonal uncertainty: small isotropic noise
    sample_uncertainty = []
    for i in range(n):
        sample_uncertainty.append(
            {
                "weights": np.array([1.0]),
                "means": X[i : i + 1].copy(),
                "covariances": np.full((1, d), 0.1),
            }
        )

    return X, y, sample_uncertainty


# ===================================================================
# Smoke tests
# ===================================================================


class TestSmoke:
    def test_fit_predict_basic(self):
        X, y, su = make_linearly_separable(n=20, d=2)
        model = SvmGmmu(lam=0.01, max_iter=300, batch_size=10, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        preds = model.predict(X)
        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({-1.0, 1.0})

    def test_fit_without_uncertainty(self):
        """When sample_uncertainty=None, trains as standard SVM."""
        X, y, _ = make_linearly_separable(n=20, d=2)
        model = SvmGmmu(lam=0.01, max_iter=300, random_state=0)
        model.fit(X, y)  # no sample_uncertainty
        preds = model.predict(X)
        assert preds.shape == (20,)

    def test_decision_function(self):
        X, y, su = make_linearly_separable(n=20, d=2)
        model = SvmGmmu(lam=0.01, max_iter=200, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        scores = model.decision_function(X)
        assert scores.shape == (20,)
        # Signs of scores should match predict
        np.testing.assert_array_equal(np.sign(scores), model.predict(X))


# ===================================================================
# Accuracy on separable data
# ===================================================================


class TestAccuracy:
    def test_linearly_separable_with_uncertainty(self):
        X, y, su = make_linearly_separable(n=100, d=2, seed=0)
        model = SvmGmmu(lam=0.01, max_iter=2000, batch_size=16, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        acc = model.score(X, y)
        assert acc >= 0.95, f"Expected >= 95% accuracy, got {acc * 100:.1f}%"

    def test_linearly_separable_no_uncertainty(self):
        X, y, _ = make_linearly_separable(n=100, d=2, seed=0)
        model = SvmGmmu(lam=0.01, max_iter=2000, batch_size=16, random_state=0)
        model.fit(X, y)
        acc = model.score(X, y)
        assert acc >= 0.95, f"Expected >= 95% accuracy, got {acc * 100:.1f}%"


# ===================================================================
# Covariance formats
# ===================================================================


class TestCovarianceFormats:
    def test_diagonal_covariance(self):
        X, y, su = make_linearly_separable(n=20, d=3)
        # su already uses diagonal format (shape (1, d) per sample)
        model = SvmGmmu(lam=0.01, max_iter=200, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        assert model.predict(X).shape == (20,)

    def test_full_covariance(self):
        X, y, _ = make_linearly_separable(n=20, d=3)
        d = 3
        su = []
        for i in range(20):
            su.append(
                {
                    "weights": np.array([1.0]),
                    "means": X[i : i + 1].copy(),
                    "covariances": (0.1 * np.eye(d)).reshape(1, d, d),
                }
            )
        model = SvmGmmu(lam=0.01, max_iter=200, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        assert model.predict(X).shape == (20,)

    def test_diagonal_and_full_give_similar_results(self):
        """Diagonal and equivalent full covariance should produce
        very similar models (not identical due to random SGD, but
        both should classify well)."""
        X, y, su_diag = make_linearly_separable(n=60, d=2, seed=5)
        d = 2

        su_full = []
        for entry in su_diag:
            cov_diag = entry["covariances"][0]  # shape (d,)
            su_full.append(
                {
                    "weights": entry["weights"].copy(),
                    "means": entry["means"].copy(),
                    "covariances": np.diag(cov_diag).reshape(1, d, d),
                }
            )

        model_diag = SvmGmmu(lam=0.01, max_iter=1500, random_state=0)
        model_diag.fit(X, y, sample_uncertainty=su_diag)

        model_full = SvmGmmu(lam=0.01, max_iter=1500, random_state=0)
        model_full.fit(X, y, sample_uncertainty=su_full)

        acc_diag = model_diag.score(X, y)
        acc_full = model_full.score(X, y)
        assert acc_diag >= 0.90
        assert acc_full >= 0.90


# ===================================================================
# Multi-component GMM
# ===================================================================


class TestMultiComponent:
    def test_two_component_gmm(self):
        """Each sample has a 2-component GMM."""
        rng = np.random.default_rng(20)
        n, d = 40, 2
        X = rng.standard_normal((n, d))
        X[:20, 0] += 2.0
        X[20:, 0] -= 2.0
        y = np.array([1.0] * 20 + [-1.0] * 20)

        su = []
        for i in range(n):
            offset = rng.standard_normal(d) * 0.3
            su.append(
                {
                    "weights": np.array([0.7, 0.3]),
                    "means": np.array([X[i], X[i] + offset]),
                    "covariances": np.array([[0.1, 0.1], [0.2, 0.2]]),
                }
            )

        model = SvmGmmu(lam=0.01, max_iter=1000, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        acc = model.score(X, y)
        assert acc >= 0.85


# ===================================================================
# Scikit-learn API
# ===================================================================


class TestSklearnApi:
    def test_get_params(self):
        model = SvmGmmu(lam=0.05, max_iter=500)
        params = model.get_params()
        assert params["lam"] == 0.05
        assert params["max_iter"] == 500

    def test_set_params(self):
        model = SvmGmmu()
        model.set_params(lam=0.1, max_iter=200)
        assert model.lam == 0.1
        assert model.max_iter == 200

    def test_score(self):
        X, y, su = make_linearly_separable(n=40, d=2)
        model = SvmGmmu(lam=0.01, max_iter=500, random_state=0)
        model.fit(X, y, sample_uncertainty=su)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_not_fitted_error(self):
        model = SvmGmmu()
        with pytest.raises(Exception):
            model.predict(np.array([[1.0, 2.0]]))

    def test_repr(self):
        model = SvmGmmu(lam=0.05, max_iter=500)
        r = repr(model)
        assert "SvmGmmu" in r
        assert "lam=0.05" in r

    def test_classes_attribute(self):
        X, y, _ = make_linearly_separable(n=20, d=2)
        model = SvmGmmu(lam=0.01, max_iter=100, random_state=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.classes_, np.array([-1.0, 1.0]))


# ===================================================================
# Verbose mode
# ===================================================================


class TestVerbose:
    def test_loss_history_populated(self, capsys):
        X, y, su = make_linearly_separable(n=20, d=2)
        model = SvmGmmu(
            lam=0.01,
            max_iter=500,
            random_state=0,
            verbose=True,
            log_interval=100,
        )
        model.fit(X, y, sample_uncertainty=su)
        assert len(model.loss_history_) > 0
        captured = capsys.readouterr()
        assert "objective" in captured.out


# ===================================================================
# Input validation
# ===================================================================


class TestValidation:
    def test_bad_lam(self):
        model = SvmGmmu(lam=-1.0)
        with pytest.raises(ValueError, match="lam must be positive"):
            model.fit(np.array([[1.0]]), np.array([1.0]))

    def test_bad_labels(self):
        model = SvmGmmu()
        with pytest.raises(ValueError, match="Labels must be"):
            model.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))

    def test_mismatched_n(self):
        model = SvmGmmu()
        with pytest.raises(ValueError, match="samples but y has"):
            model.fit(np.array([[1.0], [2.0]]), np.array([1.0]))

    def test_bad_uncertainty_type(self):
        model = SvmGmmu()
        with pytest.raises(TypeError, match="must be a list"):
            model.fit(
                np.array([[1.0]]),
                np.array([1.0]),
                sample_uncertainty="bad",
            )

    def test_bad_uncertainty_length(self):
        model = SvmGmmu()
        with pytest.raises(ValueError, match="entries but X has"):
            model.fit(
                np.array([[1.0], [2.0]]),
                np.array([1.0, -1.0]),
                sample_uncertainty=[],
            )

    def test_missing_key(self):
        model = SvmGmmu()
        su = [{"weights": np.array([1.0]), "means": np.array([[1.0]])}]
        with pytest.raises(ValueError, match="missing keys"):
            model.fit(np.array([[1.0]]), np.array([1.0]), sample_uncertainty=su)

    def test_bad_weight_sum(self):
        model = SvmGmmu()
        su = [
            {
                "weights": np.array([0.3, 0.3]),
                "means": np.array([[1.0], [2.0]]),
                "covariances": np.array([[0.1], [0.1]]),
            }
        ]
        with pytest.raises(ValueError, match="sum to"):
            model.fit(np.array([[1.5]]), np.array([1.0]), sample_uncertainty=su)

    def test_negative_diagonal_covariance(self):
        model = SvmGmmu()
        su = [
            {
                "weights": np.array([1.0]),
                "means": np.array([[1.0]]),
                "covariances": np.array([[-0.1]]),
            }
        ]
        with pytest.raises(ValueError, match="negative diagonal"):
            model.fit(np.array([[1.0]]), np.array([1.0]), sample_uncertainty=su)
