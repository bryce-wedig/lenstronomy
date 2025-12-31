# Quick test in Python
import numpy as np
import sys
from lenstronomy.LightModel.Profiles.mge_multi_set import MGEMultiSet

mge = MGEMultiSet(n_sets=2, n_gaussians_per_set=30)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
x_flat, y_flat = X.flatten(), Y.flatten()

kwargs = {
    "amp": np.ones(60),
    "sigma_min": 0.006,
    "sigma_max": 2.5,
    "center_x": 0.0,
    "center_y": 0.0,
    "e1_set0": 0.05,
    "e2_set0": -0.07,
    "e1_set1": 0.04,
    "e2_set1": 0.08,
}

# Test function_split
responses = mge.function_split(x_flat, y_flat, **kwargs)
print(f"Number of basis functions: {len(responses)}")
print(f"Sum of first basis function: {np.sum(responses[0])}")
print(f"Sum of last basis function: {np.sum(responses[-1])}")
print(f"Max value in first basis: {np.max(responses[0])}")