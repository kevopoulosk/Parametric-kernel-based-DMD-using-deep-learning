import numpy as np
import matplotlib.pyplot as plt


def Lotka_Volterra(alpha, beta, gamma, delta, T, dt, x0, y0):
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


alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1
x, y = Lotka_Volterra(alpha=alpha, beta=beta, gamma=gamma, delta=delta, T=100, dt=0.0001, x0=10, y0=10)

time = np.linspace(0, 100, int(100 / 0.0001))
plt.figure(figsize=(10,7))
plt.plot(time, x, label='prey')
plt.plot(time, y, label='predator', linestyle='--')
plt.legend(loc='best')
plt.xlabel(r"$t$")
plt.ylabel('Population')
plt.grid(True)
plt.show()
