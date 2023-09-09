import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def lorenz(state, t0, sigma = 10, beta = 8./3., rho = 28.0):
    x = state[0]
    y = state[1]
    z = state[2]
    
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    return[dxdt, dydt, dzdt]


tset = np.linspace(0,100,10000)
x0 = 10 + 0.01 * np.random.random(3)

state = odeint(lorenz, x0, tset)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(state[:,0],state[:,1],state[:,2])
plt.show()