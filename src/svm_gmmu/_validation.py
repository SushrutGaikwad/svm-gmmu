"""Input validation utilities for SVM-GMMU.

This module will contain:
- validate_sample_uncertainty: checks that the list-of-dicts structure is
  well-formed (correct shapes, weights sum to 1, covariances positive, etc.)
- infer_covariance_type: auto-detects diagonal vs full covariance from shape
"""
