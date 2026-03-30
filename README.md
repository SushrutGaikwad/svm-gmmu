# SVM-GMMU

SVM with Gaussian Mixture Model Uncertainty.

A scikit-learn-compatible classifier that accounts for per-sample uncertainty modeled as Gaussian mixtures. Includes the single-Gaussian special case (SVM-GSU) from Tzelepis, Mezaris, and Patras (IEEE TPAMI, 2017) and extends it to mixtures of Gaussians (SVM-GMMU).

## Installation

```bash
uv sync
```

For development (includes pytest and matplotlib):

```bash
uv sync --extra dev
```

## Quick start

```python
import numpy as np
from svm_gmmu import SvmGmmu

# Each sample has a GMM uncertainty description
sample_uncertainty = [
    {
        "weights": np.array([0.6, 0.4]),
        "means": np.array([[0.8, 1.5], [1.3, 2.8]]),
        "covariances": np.array([[0.1, 0.2], [0.3, 0.4]]),
    },
    {
        "weights": np.array([1.0]),
        "means": np.array([[3.0, 4.0]]),
        "covariances": np.array([[0.05, 0.05]]),
    },
]

X = np.array([[1.0, 2.0], [3.0, 4.0]])  # overall mixture means
y = np.array([+1, -1])

model = SvmGmmu(lam=0.01, max_iter=1000, batch_size=1)
model.fit(X, y, sample_uncertainty=sample_uncertainty)
predictions = model.predict(X)
```
