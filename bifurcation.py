import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar


def lorenz(x, y, z, r, b=10, s=6):
    xdot = b * (y - x)
    ydot = r * x - y - x * z
    zdot = x * y - s * z
    return xdot, ydot, zdot


# --- Setup ---
dr = 0.1  # parameter step size
r = np.arange(40, 200, dr)  # parameter range
dt = 0.001  # time step
t = np.arange(0, 10, dt)  # time range

# initialize solution arrays
xs = np.empty(len(t) + 1)
ys = np.empty(len(t) + 1)
zs = np.empty(len(t) + 1)

# initial condition
xs[0], ys[0], zs[0] = (1, 1, 1)


rmax = []
zmax = []
rmins = []
zmins = []


# --- Simulation ---
with alive_bar(len(r)) as bar:
    for R in r:

        for i in range(len(t)):
            # approximate numerical solutions to system
            xdot, ydot, zdot = lorenz(xs[i], ys[i], zs[i], R)
            xs[i + 1] = xs[i] + (xdot * dt)
            ys[i + 1] = ys[i] + (ydot * dt)
            zs[i + 1] = zs[i] + (zdot * dt)
        # calculate and save the peak values of the z solution
        for i in range(1, len(zs) - 1):
            # save the local maxima
            if zs[i - 1] < zs[i] and zs[i] > zs[i + 1]:
                rmax.append(R)
                zmax.append(zs[i])
            # save the local minima
            elif zs[i - 1] > zs[i] and zs[i] < zs[i + 1]:
                rmins.append(R)
                zmins.append(zs[i])

        # "use final values from one run as initial conditions for the next to stay near the attractor"
        xs[0], ys[0], zs[0] = xs[i], ys[i], zs[i]

        bar()


# --- Plotting ---
plt.scatter(rmax, zmax, color="black", s=0.5, alpha=0.2)
plt.scatter(rmins, zmins, color="red", s=0.5, alpha=0.2)

plt.xlim(0, 200)
plt.ylim(0, 400)
plt.show()