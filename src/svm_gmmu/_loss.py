"""Closed-form loss and gradient computations for SVM-GMMU.

This module will contain:
- d_mu: signed distance of the mean from the margin boundary
- d_sigma: uncertainty spread in the classification-relevant direction
- component_loss: closed-form expected hinge loss for a single Gaussian
- gmmu_loss: weighted sum of component losses for a GMM
- gmmu_grad_w: gradient of the objective with respect to w
- gmmu_grad_b: gradient of the objective with respect to b
"""
