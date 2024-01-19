import numpy as np
from Lotka_Volterra_model import Lotka_Volterra_Snapshot, Predict, Lotka_Volterra
from Lotka_Volterra_Deriv import Lotka_Volterra_Deriv
from Sparse_Dictionary_Learning import Permute, linear_kernel, quadratic_kernel, SparseDictionary
import matplotlib.pyplot as plt
from scipy import integrate

# Define parameters and initial condition
params = [0.1, 0.002, 0.2, 0.0025]
IC = [80, 20]

# Calculate the snapshot matrix "X", and the derivative matrix "Y"
X, _ = Lotka_Volterra_Snapshot(params, T=100, dt=0.0002, x0=IC[0], y0=IC[1])
Y = Lotka_Volterra_Deriv(X, *params)

# randomly permute X and Y, for more efficient dictionary learning
Xperm, perm = Permute(X)
Yperm = Y[:, perm]

# Learn the Sparse Dictionary for both kernels
Dict_linear, mVals_linear, deltaVals_linear = SparseDictionary(Xperm, kernel=linear_kernel, tolerance=1e-6)
Dict_quad, mVals_quad, deltaVals_quad = SparseDictionary(Xperm, kernel=quadratic_kernel, tolerance=1e-6)


# Compute W tilde for both kernels
W_tilde_linear = Yperm @ np.linalg.pinv(quadratic_kernel(Dict_linear, Xperm))
W_tilde_quad = Yperm @ np.linalg.pinv(quadratic_kernel(Dict_quad, Xperm))

# Form the model
Model_Linear = W_tilde_linear @ linear_kernel(Dict_linear, X)
Model_Quad = W_tilde_quad @ quadratic_kernel(Dict_quad, X)

recErr_linear = np.mean(np.linalg.norm(Y - Model_Linear) / np.linalg.norm(Y))
print(f"The reconstruction error using linear kernel is {recErr_linear}")

recErr_quad = np.mean(np.linalg.norm(Y - Model_Quad) / np.linalg.norm(Y))
print(f"The reconstruction error using quadratic kernel is {recErr_quad}")


# Form the general model, using the quadratic kernel
def Model_General(x, t):
    return W_tilde_quad @ quadratic_kernel(Dict_quad, x)


time = np.linspace(0, 100, 500000)
plt.figure(figsize=(10, 7))
plt.title("Reconstruction of the Lotka-Volterra system (same initial condition, same parameters)")
plt.plot(time, Y[0, :], label=r"$\dot{x0}$", color='red')
plt.plot(time, Y[1, :], label=r"$\dot{x1}$", color='green')
plt.plot(time, Model_Quad[0, :], label='Prediction quad', linestyle='-.', color='black')
plt.plot(time, Model_Quad[1, :], label='Prediction quad', linestyle='-.', color='black')

plt.plot(time, Model_Linear[0, :], label='Prediction linear', linestyle='--', color='black')
plt.plot(time, Model_Linear[1, :], label='Prediction linear', linestyle='--', color='black')
plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()


# TODO : Fix the error regarding the numerical integration. For some reason the integration blows up.
# Potentially, I handle the odeint routine wrong.
t = np.linspace(0, 100, 500000)
# Prediction using the quadratic kernel
Pred_Reconstruction = Predict(model=Model_General, Tend=100, IC=IC)
plt.title("Numerically integrate the constructed model/ Reconstruction")
plt.plot(t, Pred_Reconstruction[:, 0], label='x0')
plt.plot(t, Pred_Reconstruction[:, 1], label='x1')
plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()


# Try predicting the system with the same parameter realization, but different initial condition
IC_other = [35, 5]
X_diff_IC, _ = Lotka_Volterra_Snapshot(params, T=100, dt=0.0002, x0=IC_other[0], y0=IC_other[1])
Y_diff_IC = Lotka_Volterra_Deriv(X_diff_IC, *params)

Y_pred = W_tilde_quad @ quadratic_kernel(Dict_quad, X_diff_IC)

time = np.linspace(0, 100, 500000)
plt.figure(figsize=(10, 7))
plt.title("Predict the Lotka-Volterra system, same parameters, different initial condition")
plt.plot(time, Y_diff_IC[0, :], label=r"$\dot{x0}$", color='red')
plt.plot(time, Y_diff_IC[1, :], label=r"$\dot{x1}$", color='green')
plt.plot(time, Y_pred[0, :], label='Prediction quad', linestyle='-.', color='black')
plt.plot(time, Y_pred[1, :], label='Prediction quad', linestyle='-.', color='black')


plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()





