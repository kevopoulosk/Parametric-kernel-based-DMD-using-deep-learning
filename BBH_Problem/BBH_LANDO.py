import numpy as np
from Lotka_Volterra.Sparse_Dictionary_Learning import *
import matplotlib.pyplot as plt
from DynamicalSystem import *
from sklearn.metrics import pairwise_distances


def kernel_sine(u, v, l=0.5, p=1):
    d = pairwise_distances(u, v, metric='euclidean')
    res = np.exp(- (2 * np.sin(np.pi * d / p) ** 2) / l ** 2)
    return res


### Define parameters and initial condition
params = [100, 0.9]
IC = [0, np.pi]
T = 1e5
dt = 1

### Calculate the snapshot matrix "X", and the derivative matrix "Y"
X = BBH_Model(x_initial=IC, total_time=T, dt=dt, parameters=params)
Y = BBH_Deriv(X, *params)

scaledX = Scale(X)

### randomly permute X and Y, to improve the numerical conditioning of the dictionary learning
Xperm, perm = Permute(X)
Yperm = Y[:, perm]

### Learn the Sparse Dictionary for both kernels
Dict_gauss, mVals_gauss, deltaVals_gauss, _ = SparseDictionary(Xperm, scaledX, kernel=kernel_sine, tolerance=1e-6)

### Compute W tilde for both kernels
W_tilde_gauss = Yperm @ np.linalg.pinv(kernel_sine(Dict_gauss, scaledX * Xperm))

### Form the model
Model_Gauss = W_tilde_gauss @ kernel_sine(Dict_gauss, scaledX * X)

print(f"CONDITION NUMBER of GAUSS: {np.linalg.cond(gauss_kernel(Dict_gauss, scaledX * X))}")

recErr_gauss = np.mean(np.linalg.norm(Y - Model_Gauss) / np.linalg.norm(Y))
print(f"The reconstruction error using Gaussian kernel is {recErr_gauss}")


### Form the general model, using the quadratic kernel
def Model_General(t, z):
    x0, x1 = z
    x = np.array([[x0], [x1]])
    return (W_tilde_gauss @ gauss_kernel(Dict_gauss, scaledX * x)).flatten()


time = np.linspace(0, 1e5, int(1e5))

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.plot(time, Y[0, :], label=r"$\dot{x0}$", color='red')
plt.plot(time, Model_Gauss[0, :], label='Prediction gauss', linestyle='dotted', color='blue')
plt.plot(time, Y[1, :], label=r"$\dot{x1}$", color='green')
plt.plot(time, Model_Gauss[1, :], label='Prediction gauss', linestyle='dotted', color='blue')
plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.title("Reconstruction of Lotka-Volterra with Gauss Model")
plt.grid(True)
plt.show()

### Prediction using the quadratic kernel, for a different initial condition
Pred_Reconstruction = Predict(model=Model_General, Tend=1e5, IC=IC, type="Cont", sensors=int(1e5))
plt.title("Numerically integrate the constructed model/ Reconstruction")
plt.plot(time, Pred_Reconstruction[0, :], label='x0 pred')
plt.plot(time, X[0, :], '--', color='black', label="x0 true")
plt.plot(time, Pred_Reconstruction[1, :], label='x1 pred')
plt.plot(time, X[1, :], '--', color='black', label="x0 true")

plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()

### Predict the f(x) with the same parameter realization, but different initial condition

# X_diff_IC, _ = Lotka_Volterra_Snapshot(params, T=800, x0=IC_other[0], y0=IC_other[1])
# Y_diff_IC = Lotka_Volterra_Deriv(X_diff_IC, *params)
#
# Y_pred_quad = W_tilde_quad @ quadratic_kernel(Dict_quad, scaledX*X_diff_IC)
# Y_pred_gauss = W_tilde_gauss @ gauss_kernel(Dict_gauss, scaledX*X_diff_IC)
#
# time = np.linspace(0, 400, 300)
# plt.figure(figsize=(10, 7))
# plt.title("Predict the Lotka-Volterra system, same parameters, different initial condition")
# plt.plot(time, Y_diff_IC[0, :], label=r"$\dot{x0}$", color='black')
# plt.plot(time, Y_diff_IC[1, :], label=r"$\dot{x1}$", color='black')
# plt.plot(time, Y_pred_quad[0, :], label='Prediction quad', linestyle='-.', color='green')
# plt.plot(time, Y_pred_quad[1, :], label='Prediction quad', linestyle='-.', color='red')
#
# plt.plot(time, Y_pred_gauss[0, :], label='Prediction gauss', linestyle='dotted', color='green')
# plt.plot(time, Y_pred_gauss[1, :], label='Prediction gauss', linestyle='dotted', color='green')
#
#
# plt.legend(loc='best')
# plt.xlabel(r"$t$")
# plt.ylabel('Population')
# plt.grid(True)
# plt.show()
#
# recErr_quad_diff_IC = np.mean(np.linalg.norm(Y_diff_IC - Y_pred_quad) / np.linalg.norm(Y_diff_IC))
# print(f"The prediction error for different initial condition using quadratic kernel is {recErr_quad_diff_IC}")
#
# recErr_gauss_diff_IC = np.mean(np.linalg.norm(Y_diff_IC - Y_pred_gauss) / np.linalg.norm(Y_diff_IC))
# print(f"The prediction error for different initial condition using Gaussian kernel is {recErr_gauss_diff_IC}")
#
# print(f"CONDITION NUMBER of QUAD: {np.linalg.cond(quadratic_kernel(Dict_quad, scaledX*X_diff_IC))} IN PREDICTED CASE DIFF_IC")
