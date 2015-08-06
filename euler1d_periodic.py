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
x = arange(N) * dx
gamma = 1.4

def diffx(w):
    return (roll(w, -1) - roll(w, 1)) / (2 * dx)

def rhs(w):
    r, ru, p = w[:,0], w[:,1], w[:,2]
    u = ru / r
    rhs_w = zeros(w.shape)
    rhs_w[:,0] = 0.5 * diffx(r * ru) / r
    rhs_w[:,1] = ((diffx(ru*ru) + r*ru * diffx(u)) / 2.0 + diffx(p)) / r
    rhs_w[:,2] = gamma * diffx(p * u) - (gamma - 1) * u * diffx(p)
    return rhs_w

def midpoint_res(w1, w0):
    w = 0.5 * (w0 + w1)
    return (w1 - w0) / dt + rhs(w)

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
u = u0 * wave**0

w = zeros([N, 3])
w[:,0] = sqrt(rho)
w[:,1] = sqrt(rho) * u
w[:,2] = p

# w *= 0.9 + 0.2 * random.random(w.shape)

print(ddt_conserved(w, rhs(w)))

print(conserved(w))

for iplot in range(10):
    for istep in range(10):
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

