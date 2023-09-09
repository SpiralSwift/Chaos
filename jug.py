import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def food_chain(state, t0, a1 = 5.0 , a2 = 0.1, b1 = 3.0, b2 = 2.0, d1 = 0.4, d2 = 0.01):
    x = state[0]
    y = state[1]
    z = state[2]

    dx = x*(1-x) - (a1*x*y)/(1+b1*x)
    dy = (a1*x*y)/(1+b1*x) - (a2*y*z)/(1+b2*y) - d1*y
    dz = (a2*y*z)/(1+b2*y) - d2*z

    return [dx,dy,dz]


tset = np.linspace(0,10000,100000)
state0 = [0.8,0.3,8]

state = odeint(food_chain, state0, tset)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(state[:,0],state[:,1],state[:,2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()