import matplotlib.pyplot as plt
from pydmd import LANDO
import numpy as np
from BBH_LANDO import *


def relative_error(est, true):
    """
    Helper method for computing relative error.
    """
    return np.linalg.norm(true - est) / np.linalg.norm(true)


solve_ivp_opts = {}
solve_ivp_opts["rtol"] = 1e-12
solve_ivp_opts["atol"] = 1e-12
solve_ivp_opts["method"] = "BDF"

params = [100, 0.5]
IC = [0, np.pi]
T = 1.5e4
dt = 1
e = 0.9
p = 100
sigma = 0.2

gamma = 1 / (2 * np.square(sigma))

X = BBH_Model(x_initial=IC, total_time=T, dt=dt, parameters=params)
Y = BBH_Deriv(X, *params)

lando = LANDO(
    svd_rank=150,
    kernel_metric="rbf",
    kernel_params={"gamma": gamma},
    x_rescale=1 / np.abs(X).max(axis=1),
    dict_tol=1e-6,
)

lando.fit(X, Y)

dictionary_size = lando.sparse_dictionary.shape[-1]
Y_est = lando.f(X)
lando_error = np.round(relative_error(Y_est, Y), decimals=7) * 100

print("LANDO Training Error:       {}%".format(lando_error))
print("LANDO Dictionary Size:      {}".format(dictionary_size))

xpred = lando.predict(x0=X[:, 0], tend=int(1.5e4), continuous=True, dt=1, solve_ivp_opts=solve_ivp_opts)

err = np.round(relative_error(xpred, X), decimals=7) * 100
print(f"The prediction error of x is {err}%")
plt.plot(xpred[0], '--', label='pred')
plt.plot(X[0], label='tru')
plt.legend()
plt.show()

rt = p/(1+e*np.cos(xpred[1]))
xt = -rt*np.cos(xpred[0])
yt = -rt*np.sin(xpred[0])

rt_true = p/(1+e*np.cos(X[1]))
xt_true = -rt_true*np.cos(X[0])
yt_true = -rt_true*np.sin(X[0])

plt.plot(xt_true, yt_true, label='truth')
plt.plot(xt, yt, '--', label='pred lando')
plt.legend()
plt.show()


