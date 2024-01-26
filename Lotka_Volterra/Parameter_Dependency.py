from Sparse_Dictionary_Learning import *
from Lotka_Volterra_Deriv import *

# At this file, we generate sparse dictionaries and form the kernel-DMD model for different parameter realizations
# Our goal is to investigate whether the kernel, Wtilde (coefficient matrix) and Xtilde (sparse dictionary), change when
# mu = [alpha, beta, gamma, delta] changes.
# At first, generate parametric samples


def SelectPoints(matrix, m):

    step_size = len(matrix) // m
    m_points = matrix[::step_size]

    return m_points


alpha_range = [0.01, 0.14]
beta_range = [0.001, 0.0025]
gamma_range = [0.02, 0.3]
delta_range = [0.0015, 0.0025]

tol = 1e-3
# 5 different parametric realizations
NumSamples = 10
W_vals = []
kernel_vals = []
errors = []
shapes = []

time = np.linspace(0, 600, int(600/0.002))
# Generate sparse dictionary and form the model for all the different "mu" realizations.
for i in range(NumSamples):
    # sample a realisation of the mu = [alpha, beta, gamma, delta] vector.
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    delta = np.random.uniform(delta_range[0], delta_range[1])
    mu = [alpha, beta, gamma, delta]
    # Generate the Snapshot matrix, X and y matrices
    X, _ = Lotka_Volterra_Snapshot(params=mu)
    Y = Lotka_Volterra_Deriv(X, *mu)

    scaledX = Scale(X)

    Xperm, perm = Permute(X)
    Yperm = Y[:, perm]

    SparseDict, m_vals, deltas = SparseDictionary(X, scaledX, quadratic_kernel, tolerance=tol)

    # Compute W tilde and form the model.
    W_tilde = Yperm @ np.linalg.pinv(quadratic_kernel(SparseDict, scaledX*Xperm))
    W_vals.append(W_tilde)

    kernel = quadratic_kernel(SparseDict, scaledX*X)
    print(f"Shape of kernel: {kernel.shape}")
    for row in kernel:
        plt.plot(row, '.-')
        plt.show()

    model = W_tilde @ kernel
    kernel_vals.append(kernel)
    shapes.append(kernel.shape[0])
    # Compute the reconstruction error
    recErr = np.mean(np.linalg.norm(Y - model) / np.linalg.norm(Y))
    errors.append(recErr)
    print(f"The reconstruction error is {recErr}")

    plt.figure(figsize=(10, 7))
    plt.title("Reconstruction of the Lotka-Volterra system (same initial condition, same parameters)")
    plt.plot(time, Y[0, :], label=r"$\dot{x0}$", color='red')
    plt.plot(time, Y[1, :], label=r"$\dot{x1}$", color='green')
    plt.plot(time, model[0, :], label='Prediction quad', linestyle='-.', color='black')
    plt.plot(time, model[1, :], label='Prediction quad', linestyle='-.', color='black')
    plt.legend(loc='best')
    plt.xlabel(r"$t$")
    plt.ylabel('Population')
    plt.grid(True)
    plt.show()






