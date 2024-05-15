from Lotka_Volterra.Sparse_Dictionary_Learning import *
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from pydmd import LANDO
import time


def plot_fixed_point_results(c, L, N):
    """
    Helper function that plots the provided bias term, linear operator,
    and nonlinear operator. Visualizes fixed point analysis results.
    """
    plt.figure(figsize=(9, 2))
    plt.subplot(1, 3, 1)
    plt.title("Bias")
    plt.imshow(c, cmap="bwr", vmax=5, vmin=-5)
    plt.colorbar()
    plt.xticks([])
    plt.subplot(1, 3, 2)
    plt.title("Linear Operator")
    plt.imshow(L, cmap="bwr", vmax=0.05, vmin=-0.05)
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title("Nonlinear Forcing")
    plt.imshow(N, aspect="auto", vmax=0.2, vmin=0)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def relative_error(est, true):
    """
    Helper method for computing relative error.
    """
    return np.linalg.norm(true - est) / np.linalg.norm(true)
### This file is used just to test the prediction of LANDO( i.e. the transformation from f(x) ---> x)

dt = 0.01
### We train the LANDO up to t = T_train = 2.3
T_train = 2

### We make the prediction up to t* = T_end = 4.2
T_end = 4.5
DirectoryProcessed = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData_Processed"
train_t_instance = int(T_train / dt)
test_t_instance = int(T_end/dt)

kernel = gauss_kernel

X = np.load(DirectoryProcessed + "/sample47.npy")[:, :train_t_instance][:, :-1]
Y = np.load(DirectoryProcessed + "/sample47.npy")[:, :train_t_instance][:, 1:]

X_comp = np.load(DirectoryProcessed + "/sample47.npy")[:, :test_t_instance]

x_bar = np.zeros(X.shape[0])
sigma = 0.2

gamma = 1 / (2 * np.square(sigma))


t1 = time.time()

# Build the LANDO model.
lando = LANDO(
    svd_rank=150,
    kernel_metric="poly",
    kernel_params={"gamma": 1.0, "degree":1.0, "coef0": 1.0},
    x_rescale=1 / np.abs(X).max(axis=1),
    dict_tol=1e-8,
)

# FITTING STEP 1: Fit the model to the input data.
# This step computes the weights of the dictionary-based kernel model.
lando.fit(X, Y)

# FITTING STEP 2: Obtain diagnostics about a fixed point.
# This step computes the linear model diagnostics and nonlinear forcings.
# Use compute_A=True to explicitly compute and store the linear operator.

lando.analyze_fixed_point(x_bar, compute_A=True)

c = lando.bias
L = lando.linear
N = lando.nonlinear(X)

# Plot c, L, and N(x).
plot_fixed_point_results(c, L, N)
t2 = time.time()

print("LANDO Fitting Time: {}".format(t2 - t1))

# View the size of the sparse dictionary, and the training accuracy of the predicted model.
dictionary_size = lando.sparse_dictionary.shape[-1]
Y_est = lando.f(X)
lando_error = np.round(relative_error(Y_est, Y), decimals=7) * 100

print("LANDO Training Error:       {}%".format(lando_error))
print("LANDO Dictionary Size:      {}".format(dictionary_size))


pred = lando.predict(x0=X[:, 0], tend=test_t_instance, continuous=False)

plt.imshow(pred, cmap='inferno', vmax=0.1, vmin=0)
plt.title("prediction of x")
plt.colorbar()
plt.show()

err = np.round(relative_error(pred, X_comp), decimals=7) * 100
print(f"The prediction error of x is {err}%")

relative_error_last = np.round(relative_error(pred[:, -1], X_comp[:, -1]), decimals=7) * 100
print(f"The prediction error of x for t=1.2 is {err}%")


plt.imshow(X_comp, cmap='inferno', vmax=0.1, vmin=0)
plt.title("ground truth of x for t=5")
plt.colorbar()
plt.show()

plt.imshow(X_comp - pred, cmap='inferno', vmax=0.1, vmin=0)
plt.title("absolute error of x for t=5")
plt.colorbar()
plt.show()




