import matplotlib.pyplot as plt
import numpy as np

from Sparse_Dictionary_Learning import *
from Lotka_Volterra_Deriv import *
import deepxde as dde
from tqdm import tqdm

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
m = 1000
NumSamples = 10
W_vals = []
kernel_vals = []
shapes = []
models = []
# Generate sparse dictionary and form the model for all the different "mu" realizations.
pbar = tqdm(total=NumSamples, desc="Generation of data...")
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

    SparseDict, m_vals, deltas = SparseDictionary(Xperm, scaledX, quadratic_kernel, tolerance=tol, pbar_bool=False)

    # Compute W tilde and form the model.
    W_tilde = Yperm @ np.linalg.pinv(quadratic_kernel(SparseDict, scaledX * Xperm))
    W_vals.append(W_tilde)

    kernel = quadratic_kernel(SparseDict, scaledX * X)
    kernel_vals.append(kernel)
    shapes.append(kernel.shape[0])

    model = W_tilde @ kernel
    models.append(model[0, :])
    # Compute the reconstruction error
    recErr = np.mean(np.linalg.norm(Y - model) / np.linalg.norm(Y))
    # errors.append(recErr)
    print(f"The reconstruction error is {recErr}")

    # for k in range(kernel.shape[0]):
    #     plt.plot(SelectPoints(kernel[k, :], m=350), '.-')
    #     plt.show()

    pbar.update()
pbar.close()

# Conctenated DeepONet training phase
min_cols = np.min([mat.shape[0] for mat in kernel_vals])
# all the kernel matrices should have the same number of basis functions.
# choose the min number of basis function that a kernel matrix has, and then truncate the other matrices to this number
kernel_vals = [mat[:min_cols, :] for mat in kernel_vals]
# choose "m" points over the concatenated kernel vector-input.
kernel_vals = [SelectPoints(mat.flatten(), m) for mat in kernel_vals]

# visualise the concatenated kernels - for understanding of the kernel unified input.
for i in range(10):
    plt.plot(kernel_vals[i], '.-')
    plt.show()

models = [SelectPoints(model_i, m) for model_i in models]
Y_train_test = np.vstack(models)

inputs_branch = np.vstack(kernel_vals)
trunk = np.linspace(0, 1000, 1000).reshape(-1, 1)

X_train = (inputs_branch[:85, :], trunk)
y_train = Y_train_test[:85, :]

X_test = (inputs_branch[85:, :], trunk)
y_test = Y_train_test[85:, :]

# np.save('X_train.npy', X_train)
# np.save('X_test.npy', X_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)


data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)


# Choose a network
dim_x = 1
# control the depth and width of the branch and trunk nets.
net = dde.nn.tensorflow_compat_v1.deeponet.DeepONetCartesianProd(
    [m, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    [dim_x, 40, 40, 40, 40, 40, 40, 40, 40, 40],
    "relu",
    "Glorot normal",
)

# Define a Model
model = dde.Model(data, net)

# Compile and Train
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
losshistory, train_state = model.train(iterations=8000, display_every=50)

# Plot the loss trajectory
dde.utils.plot_loss_history(losshistory)
plt.show()

y_pred = model.predict(X_test)
print(f"y is {y_pred} with shape {y_pred.shape}")

errors = []
t = np.linspace(0, 1000, 1000)
# visualise the actual vs predicted y values. Note y = F(x) = dx/dt
for row in range(len(y_pred)):
    err = np.linalg.norm(y_test[row, :] - y_pred[row, :]) / np.linalg.norm(y_test[row, :])
    errors.append(err)

    plt.plot(t, y_test[row, :], label=r'$x_0$ true')
    plt.plot(t, y_pred[row, :], '--', label=r'$x_0$ pred')
    plt.title("Prediction with concatenated DeepONet")
    plt.xlabel("t")
    plt.ylabel('Population')
    plt.grid(True)
    plt.show()

plt.figure()
plt.plot(errors, '.-')
plt.title("Errors for each different parameter instance")
plt.xlabel("Parameter instance")
plt.ylabel("Prediction error")
plt.show()








