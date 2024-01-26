import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import odeint


@njit
def Lotka_Volterra(alpha, beta, gamma, delta, T, dt, x0, y0):
    """
    Function that numerically integrates the coupled ODEs that comprise the Lotka-Volterra model
    :param alpha: Average per capita birth rate of prey.
    :param beta: Fraction of prey caught per predator per unit time
    :param gamma: Death rate of preadators
    :param delta: Effect of the presence of prey on the predators' growth rate.
    :param T: Total time of the simulation for the Lotka-Volterra system
    :param dt: time stepping of the integration. It is carefully selected such that the derived numerical scheme
                preserves accuracy and stability.
    :param x0: initial condition --> Initial population density of prey
    :param y0: initial condition --> Initial population density of predator

    NOTE: We can also make that function work with a routine, i.e use the "odeint" routine to integrate teh ODEs
    """

    timesteps = int(T / dt)
    x = np.zeros(timesteps)
    y = np.zeros(timesteps)
    x[0] = x0
    y[0] = y0
    # Finite Differencing
    for i in range(timesteps - 1):
        x[i + 1] = x[i] + dt * (alpha * x[i] - beta * x[i] * y[i])
        y[i + 1] = y[i] + dt * (delta * x[i] * y[i] - gamma * y[i])

    return x, y


# Example of the numerically integrated ODE
alpha, beta, gamma, delta = 0.1, 0.002, 0.2, 0.0025
x, y = Lotka_Volterra(alpha=alpha, beta=beta, gamma=gamma, delta=delta, T=100, dt=0.0002, x0=80, y0=20)

time = np.linspace(0, 100, int(100 / 0.0002))
plt.figure(figsize=(10, 7))
plt.title("The Lotka-Volterra system")
plt.plot(time, x, label='prey')
plt.plot(time, y, label='predator', linestyle='--')
plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()


def Lotka_Volterra_Snapshot(params, T=600, dt=0.002, x0=80, y0=20):
    timesteps = int(T / dt)
    X = np.zeros((timesteps, 2))

    parameter_samples = []
    parameter_samples.append(params)
    alpha, beta, gamma, delta = params

    x, y = Lotka_Volterra(alpha, beta, gamma, delta, T=T, dt=dt, x0=x0, y0=y0)
    X[:, 0] = x
    X[:, 1] = y

    return X.T, parameter_samples


def Predict(model, Tend, IC):
    t = np.linspace(0, Tend, 500000)
    sol = odeint(model, IC, t)
    print(f"The solution is {sol}")
    return sol
