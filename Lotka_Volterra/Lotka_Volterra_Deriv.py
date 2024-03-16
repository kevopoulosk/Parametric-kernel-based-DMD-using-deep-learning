import numpy as np


def Lotka_Volterra_Deriv(X, alpha, beta, gamma, delta):
    """
    Function that generates the "Y" snapshot matrix, needed for the kernel-DMD.
    Note that Y = F(x) = dx/dt
    So, essentially this function evaluates the right-hand side of the Lotka-Volterra ODEs
    :param X: The state of the system.
    :param alpha: Average per capita birth rate of prey.
    :param beta: Fraction of prey caught per predator per unit time
    :param gamma: Death rate of preadators
    :param delta: Effect of the presence of prey on the predators' growth rate.

    :return: The snapshot matrix "Y"
    """

    dxdt = np.zeros(X.shape[1])
    dydt = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        dxdt[col] = alpha * X[0, col] - beta * X[0, col] * X[1, col]
        dydt[col] = delta * X[0, col] * X[1, col] - gamma * X[1, col]

    X_dot = np.vstack((dxdt, dydt))

    return X_dot

