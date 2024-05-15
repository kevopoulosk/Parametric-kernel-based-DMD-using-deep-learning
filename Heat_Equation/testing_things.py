import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit


def BBHmodel(x_initial, total_time, dt, modelparameter):
    num_T = int(total_time / dt)

    @njit
    def model_derivative(t, state, p, e):
        phi, chi = state

        chi_dynamics = (p - 2 - 2 * e * np.cos(chi)) * np.square(1 + e * np.cos(chi)) * np.sqrt(
            p - 6 - 2 * e * np.cos(chi)) / (np.square(p)) / np.sqrt(np.square(p - 2) - 4 * np.square(e))
        phi_dynamics = (p - 2 - 2 * e * np.cos(chi)) * np.square(1 + e * np.cos(chi)) / (np.power(p, 1.5)) / np.sqrt(
            np.square(p - 2) - 4 * np.square(e))

        return [phi_dynamics, chi_dynamics]

    sol = solve_ivp(model_derivative, [0, total_time], x_initial,
                    args=(modelparameter[0], modelparameter[1]),
                    method='BDF', t_eval=np.linspace(0, total_time, num_T + 1))

    return sol.y[0, :], sol.y[1, :]

### Definite parameters
e = 0.97
p = 100

phi_0 = 0
chi_0 = np.pi

dt = 1
T = 4e5

time = np.arange(0,T+(T/(T/dt))*0.1,T/(T/dt))
xlist = BBHmodel([phi_0,chi_0],T,dt,[p,e])

rt = p/(1+e*np.cos(xlist[1]))
xt = -rt*np.cos(xlist[0])
yt = -rt*np.sin(xlist[0])


plt.plot(time,xlist[0])
plt.plot(time,xlist[1])
plt.show()


plt.clf()
plt.plot(xt,yt)
plt.plot(0,0 , 'o', color='red', label='Large Black Hole')
plt.show()