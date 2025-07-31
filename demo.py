# demo implementation of methods.py
from methods import compute_optimal_mapping, compute_reconstruction_error
import numpy as np

# sample data generation
size = 5000
params = 500
X = np.random.rand(params, size)
F = np.random.randn(500, 194) @ np.random.randn(194, 500) # rank deficient forward map
Y = F @ X # sample observations

cases = ['forward', 'inverse']
reg = 1e-3

# sample implementation for forward affine case
i = 0
affine = True
reg = 1e-3
r = 20

optimal_maps = compute_optimal_mapping(X, Y, F, r, cases[i], reg, affine)
pred, err = compute_reconstruction_error(X, Y, cases[i], optimal_maps)

print(f"The error in reconstructing X was {err}")


