# SVM-GMU

SVM with Gaussian Mixture Uncertainty.

A scikit-learn-compatible classifier that accounts for per-sample uncertainty modeled as Gaussian mixtures. Includes the single-Gaussian special case (SVM-GSU) from Tzelepis et al. [[1]](#ref1) and extends it to mixtures of Gaussians (SVM-GMU).

## Report

You can read my detailed report [here](docs/reports/report_1/report.pdf).

## Examples

You can find examples of how to use the code in the [`examples`](examples/) directory.

## Installation

```bash
uv sync
```

For development (includes pytest and matplotlib):

```bash
uv sync --extra dev
```

## Quick start

### Problem setup

We have $n = 3$ training samples in a $d = 2$ dimensional feature space. Each sample $i$ (where $i \in \{1, 2, \dots, n\}$) has an observed feature vector ($\mathbf{x}_i$), a class label ($y_i$), and an uncertainty description modeled as a Gaussian mixture (`sample_uncertainty`).

The uncertainty tells the classifier: "the true location of this sample is not exactly the observed point; it could be anywhere in this cloud of probable locations." During training, the algorithm uses only the uncertainty descriptions to learn the decision boundary. The observed $\mathbf{X}$ is not used in the optimization. It follows the standard scikit-learn convention and is what you pass to `predict` afterwards.

### Notation

The notation below matches the report:

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of training samples |
| $d$ | Number of features (dimensionality) |
| $M_i$ | Number of Gaussian components for sample $i$ |
| $\pi_i^{(m)}$ | Mixing weight of the $m$-th component of sample $i$ |
| $\boldsymbol{\mu}_i^{(m)}$ | Mean vector $(d,)$ of the $m$-th component of sample $i$ |
| $\boldsymbol{\Sigma}_i^{(m)}$ | Covariance matrix $(d, d)$ of the $m$-th component of sample $i$ |

### Observed data

$\mathbf{X}$ has shape $(n, d)$. Each row $\mathbf{x}_i$ is the observed feature vector for sample $i$. $y$ has shape $(n,)$ and contains the class labels, which must be $+1$ or $-1$.

```python
import numpy as np
from svm_gmu import SvmGmu

X = np.array([
    [1.0, 2.0],   # sample 1
    [3.5, 4.0],   # sample 2
    [2.0, 0.5],   # sample 3
])
y = np.array([+1, -1, +1])
```

### Uncertainty descriptions

`sample_uncertainty` is a list of dicts, one per sample. Each dict has three keys:

- **`"weights"`**: array of shape $(M_i,)$. Mixing weights $\pi_i^{(m)} \geq 0$ that must sum to 1: $\sum_{m=1}^{M_i} \pi_i^{(m)} = 1$.
- **`"means"`**: array of shape $(M_i, d)$. Mean vector $\boldsymbol{\mu}_i^{(m)}$ of each Gaussian component.
- **`"covariances"`**: array of shape $(M_i, d, d)$ for full covariance matrices, or $(M_i, d)$ for diagonal covariances (a vector of variances per feature). Full covariance matrices $\boldsymbol{\Sigma}_i^{(m)}$ must be symmetric and positive semi-definite.

**Sample 1** ($y_1 = +1$): a two-component GMM ($M_1 = 2$). This sample's true location is bimodal: 60% chance it is near $[0.8, 1.5]$ and 40% chance near $[1.3, 2.8]$. Both components have full (non-diagonal) covariance matrices, meaning the features are correlated within each component.

```python
sample_1 = {
    "weights": np.array([0.6, 0.4]),
    "means": np.array([
        [0.8, 1.5],
        [1.3, 2.8],
    ]),
    "covariances": np.array([
        [[0.10, 0.03],
         [0.03, 0.20]],
        [[0.30, -0.05],
         [-0.05, 0.40]],
    ]),
}
```

Here, $\boldsymbol{\mu}_1^{(1)} = [0.8, 1.5]$ with a positively correlated covariance $\boldsymbol{\Sigma}_1^{(1)}$, and $\boldsymbol{\mu}_1^{(2)} = [1.3, 2.8]$ with a slightly negatively correlated $\boldsymbol{\Sigma}_1^{(2)}$.

**Sample 2** ($y_2 = -1$): a single-component GMM ($M_2 = 1$). This is the SVM-GSU special case: just one Gaussian with tight, nearly isotropic uncertainty.

```python
sample_2 = {
    "weights": np.array([1.0]),
    "means": np.array([
        [3.5, 4.0],
    ]),
    "covariances": np.array([
        [[0.05, 0.01],
         [0.01, 0.05]],
    ]),
}
```

Here $\boldsymbol{\mu}_2^{(1)} = [3.5, 4.0]$ matches the observed point, and $\boldsymbol{\Sigma}_2^{(1)}$ is small and nearly diagonal.

**Sample 3** ($y_3 = +1$): a three-component GMM ($M_3 = 3$). Complex, multimodal uncertainty.

```python
sample_3 = {
    "weights": np.array([0.5, 0.3, 0.2]),
    "means": np.array([
        [2.0, 0.5],
        [1.5, 0.8],
        [2.5, 0.2],
    ]),
    "covariances": np.array([
        [[0.08, 0.02],
         [0.02, 0.06]],
        [[0.12, -0.03],
         [-0.03, 0.10]],
        [[0.05, 0.00],
         [0.00, 0.15]],
    ]),
}

sample_uncertainty = [sample_1, sample_2, sample_3]
```

The third component $\boldsymbol{\Sigma}_3^{(3)}$ is diagonal (zero off-diagonal entries), meaning its features are uncorrelated.

### Training

Create the model and call `fit`. The hyperparameters are:

- **`lam`**: regularization strength ($\lambda$ in the report). Larger values produce a wider margin at the cost of more training errors.
- **`max_iter`**: number of SGD iterations ($T$ in the report).
- **`batch_size`**: samples per mini-batch ($k$ in the report). Set to 1 for pure stochastic gradient descent.
- **`random_state`**: seed for reproducibility.

```python
model = SvmGmu(lam=0.01, max_iter=1000, batch_size=1, random_state=42)
model.fit(X, y, sample_uncertainty=sample_uncertainty)
```

`fit` learns the weight vector $\mathbf{w}$ and bias $b$ by minimizing the SVM-GMU objective (Eq. 48 in the report):

$$\mathcal{J}(\mathbf{w}, b) = \frac{\lambda}{2}\|\mathbf{w}\|^2 + \frac{1}{n}\sum_{i=1}^{n}\sum_{m=1}^{M_i} \pi_i^{(m)}\,\mathcal{L}_i^{(m)}(\mathbf{w}, b)$$

where $\mathcal{L}_i^{(m)}$ is the closed-form expected hinge loss for the $m$-th Gaussian component of sample $i$.

### Prediction

`predict` classifies points using the learned hyperplane: $\hat{y} = \mathrm{sign}(\mathbf{w}^\intercal \mathbf{x} + b)$. No uncertainty is used at prediction time.

```python
predictions = model.predict(X)
print("Predictions:", predictions)   # expected: [+1, -1, +1]
```

`decision_function` returns the raw signed distances to the decision boundary ($\mathbf{w}^\intercal \mathbf{x} + b$), useful for ranking or thresholding.

```python
scores = model.decision_function(X)
print("Scores:", scores)
```

### Standard SVM mode (no uncertainty)

When no uncertainty information is available, omit `sample_uncertainty` (or pass `None`). The model internally treats each row of $\mathbf{X}$ as a point mass ($\boldsymbol{\Sigma}_i \to \mathbf{0}$, single-component GMM) and reduces to a standard linear SVM:

```python
model = SvmGmu(lam=0.01, max_iter=1000)
model.fit(X, y)  # no uncertainty -> standard SVM
predictions = model.predict(X)
```

## Visualization

The `svm_gmu.plotting` module provides three functions for visualizing 2-D datasets. Matplotlib is required (installed automatically with `uv sync`).

```python
from svm_gmu.plotting import plot_uncertainty, plot_boundary, plot_boundary_comparison
```

### Plot uncertainty contours (data only)

Visualize the per-sample GMM density contours before training any model. Each sample's uncertainty is shown as nested filled contours at the requested sigma levels, colored by class label, with the observed points overlaid.

```python
fig, ax = plot_uncertainty(X, y, sample_uncertainty)
```

Customize the sigma levels, figure size, or pass your own axes:

```python
fig, ax = plot_uncertainty(
    X, y, sample_uncertainty,
    sigmas=(1, 2, 3, 4),
    figsize=(12, 12),
    title="My Dataset",
    random_state=42,
)
```

### Plot a decision boundary

After fitting a model, overlay its decision boundary ($\mathbf{w}^\intercal \mathbf{x} + b = 0$, solid line) and margin lines ($\pm 1$, dashed) on top of the uncertainty contours.

```python
model = SvmGmu(lam=0.01, max_iter=5000, batch_size=1, random_state=42)
model.fit(X, y, sample_uncertainty=sample_uncertainty)

fig, ax = plot_boundary(X, y, sample_uncertainty, model)
```

### Compare SVM-GMU vs. standard SVM

Train both an uncertainty-aware model and a standard SVM, then plot them side by side to see how uncertainty shifts the decision boundary.

```python
model_gmu = SvmGmu(lam=0.01, max_iter=5000, batch_size=1, random_state=42)
model_gmu.fit(X, y, sample_uncertainty=sample_uncertainty)

model_svm = SvmGmu(lam=0.01, max_iter=5000, batch_size=1, random_state=42)
model_svm.fit(X, y)  # no uncertainty -> standard SVM

fig, (ax_left, ax_right) = plot_boundary_comparison(
    X, y, sample_uncertainty,
    model_gmu, model_svm,
)
```

All three functions accept a `random_state` parameter for reproducible sigma-level estimation and return the matplotlib `Figure` and `Axes` objects for further customization.

## References

<a id="ref1"></a>
[1] C. Tzelepis, V. Mezaris, and I. Patras, "Linear Maximum Margin Classifier for Learning from Uncertain Data," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 12, pp. 2948–2262, 2017. [doi.org/10.1109/TPAMI.2017.2772235](https://doi.org/10.1109/TPAMI.2017.2772235)
