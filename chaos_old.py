
import numpy as np
import matplotlib.pyplot as plt


# --- Paramters ---
tf = 10000 # end time
res = 200000 # resolution (number of time points)
dt = tf/res # time step

xo = 1 # initial phytoplankton population
yo = 1 # initial zooplankton population
zo = 1 # initial fish population

# non-dimensional parameters
a1 = 5.0 
a2 = 0.1
b1 = 3.0
b2 = 2.0
d1 = 0.4
d2 = 0.01

# non-dimensional parameters (parallel system)
a1p = 5.0 
a2p = 0.1
b1p = 3.0
b2p = 2.0
d1p = 0.4
d2p = 0.01

# L-V competition
alpha = 0 
beta = 0

# dimensional parameters
Ko = 1
Ro = 1
C1 = 1
C2 = 1


# --- Setup ---
print('Initializing...')
# define storage vectors
tset= np.linspace(0,tf,res)
x = np.zeros(res)
y = np.zeros(res)
z = np.zeros(res)
xp = np.zeros(res)
yp = np.zeros(res)
zp = np.zeros(res)


# seed system
x[0] = xo / Ko
y[0] = yo * C1/Ko
z[0] = zo * C1/(C2*Ko)
xp[0] = xo / Ko
yp[0] = yo * C1/Ko
zp[0] = zo * C1/(C2*Ko)


# define functions
def ev_x(xt,yt,a1,b1):
    return xt*(1-xt) - (a1*xt*yt)/(1+b1*xt) 

def ev_y(xt,yt,zt,a1,a2,b1,b2,d1):
    return (a1*xt*yt)/(1+b1*xt) - (a2*yt*zt)/(1+b2*yt) - d1*yt

def ev_z(yt,zt,a2,b2,d2):
    return (a2*yt*zt)/(1+b2*yt) - d2*zt

# with compeition
def ev_x_p(xt1,xt2,yt1,a1,b1,cparam):
    return xt1*(1-xt1-cparam*xt2) - (a1*xt1*yt1)/(1+b1*xt1)

def update(pop1,pop2,) -> np.ndarray:
    pass


# run simulation for parallel systems with competition
print('Simulating...')

for i in range(1,len(tset)):
    dt = tset[i] - tset[i-1]

    dxdt = ev_x_p(x[i-1],xp[i-1],y[i-1],a1,b1,beta)
    dydt = ev_y(x[i-1],y[i-1],z[i-1],a1,a2,b1,b2,d1)
    dzdt = ev_z(y[i-1],z[i-1],a2,b2,d2)

    dxdtp = ev_x_p(xp[i-1],x[i-1],yp[i-1],a1,b1,alpha)
    dydtp = ev_y(x[i-1],y[i-1],z[i-1],a1,a2,b1,b2,d1)
    dzdtp = ev_z(y[i-1],z[i-1],a2,b2,d2)

    x[i] = x[i-1] + dxdt*dt
    y[i] = y[i-1] + dydt*dt
    z[i] = z[i-1] + dzdt*dt

    xp[i] = xp[i-1] + dxdtp*dt
    yp[i] = yp[i-1] + dydtp*dt
    zp[i] = zp[i-1] + dzdtp*dt


# run simulation (Runge-Kutta method)
for i in range(1,len(tset)):
    h = tset[i] - tset[i-1]

    x1 = x[i-1]
    y1 = y[i-1]
    z1 = z[i-1]
    kx1 = ev_x(x1,y1,a1,b1)
    ky1 = ev_y(x1,y1,z1,a1,a2,b1,b2,d1)
    kz1 = ev_z(y1,z1,a1,b2,d2)

    x2 = x[i-1] + h * kx1/2
    y2 = y[i-1] + h * ky1/2
    z2 = z[i-1] + h * kz1/2
    kx2 = ev_x(x2,y2,a1,b1)
    ky2 = ev_y(x2,y2,z2,a1,a2,b1,b2,d1)
    kz2 = ev_z(y2,z2,a1,b2,d2)

    x3 = x[i-1] + h * kx2/2
    y3 = y[i-1] + h * ky2/2
    z3 = z[i-1] + h * kz2/2
    kx3 = ev_x(x3,y3,a1,b1)
    ky3 = ev_y(x3,y3,z3,a1,a2,b1,b2,d1)
    kz3 = ev_z(y3,z3,a1,b2,d2)

    x4 = x[i-1] + h * kx3
    y4 = y[i-1] + h * ky3
    z4 = z[i-1] + h * kz3
    kx4 = ev_x(x4,y4,a1,b1)
    ky4 = ev_y(x4,y4,z4,a1,a2,b1,b2,d1)
    kz4 = ev_z(y4,z4,a1,b2,d2)

    x[i] = x[i-1] + h/6 * (kx1 + 2*kx2 + 2*kx3 + kx4)
    y[i] = y[i-1] + h/6 * (ky1 + 2*ky2 + 2*ky3 + ky4)
    z[i] = z[i-1] + h/6 * (kz1 + 2*kz2 + 2*kz3 + kz4)


# extract actual population values
Tset = tset / Ro
Xset = x * Ko
Yset = y * Ko/C1
Zset = z * C2*Ko/C1


# plot
print('Plotting...')
fig = plt.figure()
ax0 = fig.add_subplot(projection='3d')
ax0.scatter(x,y,z,c='b')
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(xp,yp,zp,c='b')
plt.show()