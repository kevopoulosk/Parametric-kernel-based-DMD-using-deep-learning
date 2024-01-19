from Sparse_Dictionary_Learning import *
from Lotka_Volterra_Deriv import *

# At this file, we generate sparse dictionaries and form the kernel-DMD model for different parameter realizations
# Our goal is to investigate whether the kernel, Wtilde (coefficient matrix) and Xtilde (sparse dictionary), change when
# mu = [alpha, beta, gamma, delta] changes.
# At first, generate parametric samples

alpha_range = [0.01, 0.14]
beta_range = [0.001, 0.0025]
gamma_range = [0.02, 0.3]
delta_range = [0.0015, 0.0025]

tol = 1e-6
# 5 different parametric realizations
NumSamples = 5
W_vals = []
kernel_vals = []
errors = []
# Generate sparse dictionary and form the model for all the different "mu" realizations.
for i in range(NumSamples):
    # sample a realisation of the mu = [alpha, beta, gamma, delta] vector.
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    delta = np.random.uniform(delta_range[0], delta_range[1])
    mu = [alpha, beta, gamma, delta]
    # Generate the Snapshot matrix, X and y matrices
    SnapshotMat, _ = Lotka_Volterra_Snapshot(params=mu)

    X, _ = Permute(SnapshotMat)
    SparseDict, m_vals, deltas = SparseDictionary(X, quadratic_kernel, tolerance=tol)

    Y = Lotka_Volterra_Deriv(X, *mu)
    # Compute W tilde and form the model.
    W_tilde = Y @ np.linalg.pinv(quadratic_kernel(SparseDict, X))
    W_vals.append(W_tilde)

    kernel = quadratic_kernel(SparseDict, X)
    model = W_tilde @ kernel
    kernel_vals.append(kernel)
    # Compute the reconstruction error
    recErr = np.mean(np.linalg.norm(Y - model) / np.linalg.norm(Y))
    errors.append(recErr)
    print(f"The reconstruction error is {recErr}")
