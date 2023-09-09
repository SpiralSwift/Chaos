import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


def animate_func(i, states, xlim, ylim, zlim):
    axs = [ax1,ax2,ax3,ax4]
    for ax in axs:
        ax.clear()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    ax1.plot3D(states[i][0][0,:],states[i][0][1,:],states[i][0][2,:])
    ax2.plot3D(states[i][0][3,:],states[i][0][4,:],states[i][0][5,:])
    ax3.plot3D(states[i][1][0,:],states[i][1][1,:],states[i][1][2,:])
    ax4.plot3D(states[i][1][3,:],states[i][1][4,:],states[i][1][5,:])

def flatten(a) -> list:
    return [item for sublist in a for item in sublist]

def dxdt(x1,y1,x2,a1,b1,alpha):
    return x1*(1-x1-alpha*x2) - (a1*x1*y1)/(1+b1*x1)

def dydt(x,y,z,a1,b1,d1,a2,b2):
    return (a1*x*y)/(1+b1*x) - (a2*y*z)/(1+b2*y) - d1*y

def dzdt(y,z,a2,b2,d2):
    return (a2*y*z)/(1+b2*y) - d2*z

def paired_food_chains(t, state,  a1 = 5.0, a2 = 0.1, b1a = 3.0, b1b = 2.0, b2 = 2.0, d1 = 0.4, d2 = 0.01, alpha = 0.0, beta = 0.0):
    x = [state[0],state[3]]
    y = [state[1],state[4]]
    z = [state[2],state[5]]
    c = [alpha,beta]
    b1 = [b1a,b1b]

    delta = []
    for i in range(2):
        j = 1 if i == 0 else 0
        dx = dxdt(x[i],y[i],x[j],a1,b1[i],c[i])
        dy = dydt(x[i],y[i],z[i],a1,b1[i],d1,a2,b2)
        dz = dzdt(y[i],z[i],a2,b2,d2)
        delta.append([dx,dy,dz])

    return np.asarray(flatten(delta))

def runge_kutta(func, tset:np.ndarray, state0:np.ndarray, **kwargs):
    state = [state0]
    npts = len(tset)
    for i in range(1,npts):
        h = tset[i] - tset[i-1]

        s1 = state[i-1]
        k1 = func(i,s1,**kwargs)
        
        s2 = s1 + h*k1/2
        k2 = func(i,s2,**kwargs)

        s3 = s1 + h*k2/2
        k3 = func(i,s3,**kwargs)

        s4 = s1 + h*k3
        k4 = func(i,s4,**kwargs)

        sout = s1 + h/6*(k1 + 2*k2 + 2*k3 + k4)
        sout = [0 if i < 0 else i for i in sout]
        state.append(sout)

        if 0 in sout:
            for j in range(len(state),npts):
                state.append(np.zeros(state0.shape))
            break

    return np.asarray(state).T


# --- Parameters ---
tset = np.linspace(0,10000,100000)
rt = 1/5 # fraction of data points to ignore
stable = [0.75,0.125,13.75] # stable population
state0 = np.asarray([0.75,0.125,13.75,0.75,0.125,13.75]) # initial state
alpha = np.linspace(0,1,100) # competition parameter
idx = [0,3] # parameters of interest


# --- Setup ---
print("Initializing...")
# initialize data vectors
print("> test states")
xmin = [[[] for i in range(2)]for j in range(2)]
xmax = [[[] for i in range(2)]for j in range(2)]
ymin = [[[] for i in range(2)]for j in range(2)]
ymax = [[[] for i in range(2)]for j in range(2)]
colmin = [[[] for i in range(2)]for j in range(2)]
colmax = [[[] for i in range(2)]for j in range(2)]

# create events
def ev0(t, y, *args): return y[0]
def ev1(t, y, *args): return y[1]
def ev2(t, y, *args): return y[2]
def ev3(t, y, *args): return y[3]
def ev4(t, y, *args): return y[4]
def ev5(t, y, *args): return y[5]
eventList = [ev0,ev1,ev2,ev3,ev4,ev5]
for ev in eventList:
    ev.terminal = True
    ev.direction = -1
events = tuple(i for i in eventList)


# --- Simulation ---
print("Simulating...")
print("> default state")
state = runge_kutta(paired_food_chains, tset, state0)
#state = solve_ivp(paired_food_chains, [tset[0],tset[-1]], state0, args=(5.0,0.1,3.0,2.0,2.0,0.4,0.01,0.0,0.0), events=events) # default case
#state = np.asarray(state.y)

print("> experimental states")
expstates = []
with alive_bar(len(alpha)) as bar:
    for a in alpha:
        #state1 = solve_ivp(paired_food_chains, [tset[0],tset[-1]], state0, args=(5.0,0.1,3.0,2.0,2.0,0.4,0.01,a,0.0), events=events)
        #state2 = solve_ivp(paired_food_chains, [tset[0],tset[-1]], state0, args=(5.0,0.1,3.0,2.0,2.0,0.4,0.01,0.0,a), events=events)
        #states = [state1.y,state2.y]
        state1 = runge_kutta(paired_food_chains, tset, state0, alpha=a)
        state2 = runge_kutta(paired_food_chains, tset, state0, beta=a)
        states = [state1,state2]
        expstates.append(states)

        for i in range(2): # system
            for j in range(2): # parameter
                t0 = int(rt*len(states[i]))
                maxvals = flatten(argrelextrema(states[i][idx[j],t0:],np.greater))
                minvals = flatten(argrelextrema(states[i][idx[j],t0:],np.less))
                col = 'r' if 0 in states[i] else 'b'  # signal extinctions
                for val in maxvals:
                    xmax[i][j].append(a)
                    ymax[i][j].append(states[i][idx[j],val])
                    colmax[i][j].append(col)
                for val in minvals:
                    xmin[i][j].append(a)
                    ymin[i][j].append(states[i][idx[j],val])
                    colmin[i][j].append(col)


        bar()


# --- Plotting ---
print("Plotting...")

# default 3D plot
fig1 = plt.figure()

ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
ax1.plot(state[0,:],state[1,:],state[2,:])
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("System 1")

ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
ax2.plot(state[3,:],state[4,:],state[5,:])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")
ax2.set_title("System 2")

plt.show()


# bifurcation diagrams (max)
fig2 = plt.figure()
ct = 0
for i in range(2): # system
    for j in range(2): # parameter
        ct += 1
        ax = fig2.add_subplot(2, 2, ct)
        ax.scatter(xmax[i][j],ymax[i][j],s=0.01,c=colmax[i][j])
plt.show()


# bifurcation diagrams (min)
fig0 = plt.figure()
ct = 0
for i in range(2): # system
    for j in range(2): # parameter
        ct += 1
        ax = fig0.add_subplot(2, 2, ct)
        ax.scatter(xmin[i][j],ymin[i][j],s=0.01,c=colmin[i][j])
plt.show()


# 3D plots
fig3 = plt.figure()

xlim = [0,max(state[0,:])]
ylim = [0,max(state[1,:])]
zlim = [0,max(state[2,:])]

ax1 = fig3.add_subplot(2, 2, 1, projection='3d')
ax2 = fig3.add_subplot(2, 2, 2, projection='3d')
ax3 = fig3.add_subplot(2, 2, 3, projection='3d')
ax4 = fig3.add_subplot(2, 2, 4, projection='3d')
ani = animation.FuncAnimation(fig3, animate_func, frames = len(expstates), fargs = [expstates,xlim,ylim,zlim])

plt.show()