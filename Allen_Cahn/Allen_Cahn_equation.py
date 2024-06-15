import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Allen_Cahn_eq(D, a, L=2.0, T=1.0, nx=250, nt=1000, plot=False):
    """
    Function that implements the discretization of the Allen-Cahn equation
    This is used for the numerical solution of the PDE. The finite differencing method is used.
    :param D: Diffusion coefficient
    :param a: Second parameter of the system
    :param L: Length of the domain
    :param T: Time horizon of the simulation
    :param nx: spatial discretization
    :param nt: time discretization
    :param plot: Boolean: Whether or not to plot specific results
    :return:
    """
    dx = L / (nx - 1)
    dt = T / nt
    # Stability condition for explicit method
    if dt > dx ** 2 / (2 * D):
        print("Warning: The time step is too large for stability. CFL Condition")

    # Initial condition
    x = np.linspace(-1, 1, nx)
    u = x ** 2 * np.cos(np.pi * x)

    # Store the solution at each time step for animation
    u_all = [u.copy()]

    if plot:
        u_all_plot = [u.copy()]
    time_steps = [0]  # Store time steps for each frame

    # Time-stepping loop
    for n in range(1, nt + 1):
        u_new = np.zeros(nx)
        # Apply periodic boundary conditions for u
        for i in range(1, nx - 1):
            u_new[i] = u[i] + dt * (D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx ** 2 - a * u[i] ** 3 + a * u[i])
        # Periodic boundary conditions for the function values
        u_new[0] = u[0] + dt * (D * (u[-2] - 2 * u[0] + u[1]) / dx ** 2 - a * u[0] ** 3 + a * u[0])
        u_new[-1] = u[-1] + dt * (D * (u[-2] - 2 * u[-1] + u[1]) / dx ** 2 - a * u[-1] ** 3 + a * u[-1])

        # Update solution
        u = u_new
        u_all.append(u.copy())
        if plot:
            # Store the new solution and time step
            if n % (nt // 100) == 0:  # Store fewer frames for faster animation
                u_all_plot.append(u.copy())
                time_steps.append(n * dt)

    X_snapshot_mat = np.vstack(u_all).T

    if plot:
        # Create an animation
        fig, ax = plt.subplots()
        line, = ax.plot(x, u_all_plot[0])
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        ax.set_xlim(-1, 1)
        ax.set_ylim(np.min(u_all_plot), np.max(u_all_plot))
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Allen-Cahn Equation Solution')

        def update(frame):
            line.set_ydata(u_all_plot[frame])
            time_text.set_text(f'Time: {time_steps[frame]:.2f}')
            return line, time_text

        ani = animation.FuncAnimation(fig, update, frames=len(u_all_plot), blit=True)
        plt.show()

    return X_snapshot_mat
