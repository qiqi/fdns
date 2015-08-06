import matplotlib
matplotlib.interactive(True)

import time
from pylab import *
from numpy import *
from numpad import *

rho0, p0, u0 = 1.2754, 101325., 100.
L = 10.
dx = 0.01
dt = 5 * dx / u0
N = int(L / dx)
x = arange(N) * dx + 0.5 * dx
gamma = 1.4

def diffx(w):
    return (w[2:] - w[:-2]) / (2 * dx)

def rhs(w):
    r, ru, p = w[:,0], w[:,1], w[:,2]
    u = ru / r
    rhs_w = zeros(w[1:-1].shape)
    rhs_w[:,0] = 0.5 * diffx(r * ru) / r[1:-1]
    rhs_w[:,1] = ((diffx(ru*ru) + (r*ru)[1:-1] * diffx(u)) / 2.0 \
                + diffx(p)) / r[1:-1]
    rhs_w[:,2] = gamma * diffx(p * u) - (gamma - 1) * u[1:-1] * diffx(p)
    return rhs_w

def apply_bc(w):
    w_ext = zeros([w.shape[0] + 2, w.shape[1]])
    w_ext[1:-1] = w
    w_ext[0,:] = [w[0,0], -w[0,1], w[0,2]]
    w_ext[-1,:] = [w[-1,0], -w[-1,1], w[-1,2]]
    return w_ext

def midpoint_res(w1, w0):
    w_ext = apply_bc(0.5 * (w0 + w1))
    return (w1 - w0) / dt + rhs(w_ext)

def conserved(w):
    r, ru, p = w[:,0], w[:,1], w[:,2]
    rho, u = r * r, ru / r
    mass = rho.sum()
    momentum = (rho * u).sum()
    energy = (p / (gamma - 1) + 0.5 * ru * ru).sum()
    return mass, momentum, energy

def ddt_conserved(w, rhs_w):
    r, ru, p = w[:,0], w[:,1], w[:,2]
    rho, u = r * r, ru / r
    ddt_rho = -rhs_w[:,0] * 2 * r
    ddt_rhou = -rhs_w[:,1] * r + 0.5 * u * ddt_rho
    ddt_p = -rhs_w[:,2]
    ddt_rhou2 = 2 * u * ddt_rhou - u**2 * ddt_rho

    ddt_mass = ddt_rho.sum()
    ddt_momentum = ddt_rhou.sum()
    ddt_energy = ddt_p.sum() / (gamma - 1) + 0.5 * ddt_rhou2.sum()
    return ddt_mass, ddt_momentum, ddt_energy

wave = 1 + 0.1 * sin(x / L * pi)**16
rho = rho0 * wave
p = p0 * wave**gamma
u = u0 * zeros(wave.shape)

w = zeros([N, 3])
w[:,0] = sqrt(rho)
w[:,1] = sqrt(rho) * u
w[:,2] = p

# w *= 0.9 + 0.2 * random.random(w.shape)

print(ddt_conserved(w, rhs(apply_bc(w))))

print(conserved(w))

for iplot in range(10):
    for istep in range(10):
        # w = solve(midpoint_res, w, (w,))
        w = solve(midpoint_res, w, (w,), verbose=False)
        w.obliviate()
    r, ru, p = w[:,0], w[:,1], w[:,2]
    rho, u = r * r, ru / r
    cla()
    plot(rho / rho0)
    plot(u / u0)
    plot(p / p0)
    draw()
    time.sleep(0.5)
    print(conserved(w))

