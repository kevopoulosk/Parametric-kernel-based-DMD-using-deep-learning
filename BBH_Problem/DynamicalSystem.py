import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


@njit
def model_derivative(t, state, p, e):
    phi, chi = state

    chi_dynamics = (p - 2 - 2 * e * np.cos(chi)) * np.square(1 + e * np.cos(chi)) * np.sqrt(
        p - 6 - 2 * e * np.cos(chi)) / (np.square(p)) / np.sqrt(np.square(p - 2) - 4 * np.square(e))
    phi_dynamics = (p - 2 - 2 * e * np.cos(chi)) * np.square(1 + e * np.cos(chi)) / (np.power(p, 1.5)) / np.sqrt(
        np.square(p - 2) - 4 * np.square(e))

    return [phi_dynamics, chi_dynamics]


def BBH_Model(x_initial, total_time, dt, parameters):

    num_T = int(total_time / dt)
    X = np.zeros((num_T, 2))

    sol = solve_ivp(model_derivative, [0, total_time], x_initial,
                    args=(parameters[0], parameters[1]),
                    method='BDF', t_eval=np.linspace(0, total_time, num_T))

    X[:, 0] = sol.y[0, :]
    X[:, 1] = sol.y[1, :]

    return X.T


@njit
def BBH_Deriv(X, p, e):
    dchi_dt = np.zeros(X.shape[1])
    dphi_dt = np.zeros(X.shape[1])

    for col in range(X.shape[1]):
        dchi_dt[col] = (p - 2 - 2 * e * np.cos(X[0, col])) * np.square(1 + e * np.cos(X[0, col])) * np.sqrt(
            p - 6 - 2 * e * np.cos(X[0, col])) / (np.square(p)) / np.sqrt(np.square(p - 2) - 4 * np.square(e))

        dphi_dt[col] = (p - 2 - 2 * e * np.cos(X[0, col])) * np.square(1 + e * np.cos(X[0, col])) / (
            np.power(p, 1.5)) / np.sqrt(
            np.square(p - 2) - 4 * np.square(e))

    X_dot = np.vstack((dchi_dt, dphi_dt))

    return X_dot
