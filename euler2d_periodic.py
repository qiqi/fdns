import matplotlib
matplotlib.interactive(True)

import time
from pylab import *
from numpy import *
from numpad import *

gamma, R = 1.4, 287.

def T_ratio(M2):
    return 1 + (gamma - 1) / 2 * M2

def p_ratio(M2):
    return T_ratio(M2)**(gamma / (gamma - 1))

T0, p0, M0 = 300., 101325., 0.2

pt_in = p0 * p_ratio(M0**2)
Tt_in = T0 * T_ratio(M0**2)
rhot_in = pt_in / (R * Tt_in)
ct_in = sqrt(gamma * R * Tt_in)

p_out = p0
c_out = sqrt(gamma * p_out / rhot_in)

rho0 = p0 / (R * T0)
c0 = sqrt(gamma * p0 / rho0)
u0 = c0 * M0

Lx, Ly = 10., 5.
dx = dy = 0.2
dt = dx / u0
Nx, Ny = int(Lx / dx), int(Ly / dy)
x = arange(Nx) * dx + 0.5 * dx
y = arange(Ny) * dy + 0.5 * dy

def diffx(w):
    return (w[2:,1:-1] - w[:-2,1:-1]) / (2 * dx)

def diffy(w):
    return (w[1:-1,2:] - w[1:-1,:-2]) / (2 * dy)

def rhs(w):
    r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    u, v = ru / r, rv / r

    rhs_w = zeros(w[1:-1,1:-1].shape)
    rhs_w[:,:,0] = 0.5 * (diffx(r * ru) + diffy(r * rv)) / r[1:-1,1:-1]
    rhs_w[:,:,1] = ((diffx(ru*ru) + (r*ru)[1:-1,1:-1] * diffx(u)) / 2.0 \
                  + (diffy(rv*ru) + (r*rv)[1:-1,1:-1] * diffy(u)) / 2.0 \
                  + diffx(p)) / r[1:-1,1:-1]
    rhs_w[:,:,2] = ((diffx(ru*rv) + (r*ru)[1:-1,1:-1] * diffx(v)) / 2.0 \
                  + (diffy(rv*rv) + (r*rv)[1:-1,1:-1] * diffy(v)) / 2.0 \
                  + diffy(p)) / r[1:-1,1:-1]
    rhs_w[:,:,-1] = gamma * (diffx(p * u) + diffy(p * v)) \
            - (gamma - 1) * (u[1:-1,1:-1] * diffx(p) + v[1:-1,1:-1] * diffy(p))
    return rhs_w

def apply_bc(w):
    w_ext = zeros([w.shape[0] + 2, w.shape[1] + 2, w.shape[-1]])
    w_ext[1:-1,1:-1] = w

    # inlet
    w_ext[0,1:-1] = w[-1,:]

    # outlet
    w_ext[-1,1:-1] = w[0,:]

    # upper wall
    w_ext[:,0] = w_ext[:,-2]

    # lower wall
    w_ext[:,-1] = w_ext[:,1]

    return w_ext

def midpoint_res(w1, w0):
    w_ext = apply_bc(0.5 * (w0 + w1))
    return (w1 - w0) / dt + rhs(w_ext)

def conserved(w):
    r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    rho, u, v = r * r, ru / r, rv / r

    mass = rho.sum()
    momentum_x = (rho * u).sum()
    momentum_y = (rho * v).sum()
    energy = (p / (gamma - 1) + 0.5 * ru * ru + 0.5 * rv * rv).sum()
    return mass, momentum_x, momentum_y, energy

def ddt_conserved(w, rhs_w):
    r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    rho, u, v = r * r, ru / r, rv / r

    ddt_rho = -rhs_w[:,:,0] * 2 * r
    ddt_rhou = -rhs_w[:,:,1] * r + 0.5 * u * ddt_rho
    ddt_rhov = -rhs_w[:,:,2] * r + 0.5 * v * ddt_rho
    ddt_p = -rhs_w[:,:,-1]
    ddt_rhou2 = 2 * u * ddt_rhou - u**2 * ddt_rho
    ddt_rhov2 = 2 * v * ddt_rhov - v**2 * ddt_rho

    ddt_mass = ddt_rho.sum()
    ddt_momentum_x = ddt_rhou.sum()
    ddt_momentum_y = ddt_rhov.sum()
    ddt_energy = ddt_p.sum() / (gamma - 1) \
            + 0.5 * ddt_rhou2.sum() + 0.5 * ddt_rhov2.sum()
    return ddt_mass, ddt_momentum_x, ddt_momentum_y, ddt_energy

wave = 1 + 0.1 * outer(sin(x / Lx * pi)**32, sin(y / Ly * pi)**8)
rho = rho0 * wave
p = p0 * wave**gamma
u = u0 * ones(wave.shape)
v = u0 * ones(wave.shape)

w = zeros([Nx, Ny, 4])
w[:,:,0] = sqrt(rho)
w[:,:,1] = sqrt(rho) * u
w[:,:,2] = sqrt(rho) * v
w[:,:,-1] = p

w *= 0.9 + 0.2 * random.random(w.shape)

print(ddt_conserved(w, rhs(apply_bc(w))))

print(conserved(w))

for iplot in range(50):
    for istep in range(1):
        # w = solve(midpoint_res, w, (w,))
        w = solve(midpoint_res, w, (w,), verbose=False)
        w.obliviate()
    print(conserved(w))
    # r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    # rho, u, v = r * r, ru / r, rv / r
    # cla()
    # subplot(2,2,1); contour(x, y, rho.T / rho0, 50); colorbar()
    # subplot(2,2,2); contour(x, y, u.T / u0, 50); colorbar()
    # subplot(2,2,3); contour(x, y, v.T / u0, 50); colorbar()
    # subplot(2,2,4); contour(x, y, p.T / p0, 50); colorbar()
    # draw()
    # time.sleep(0.5)

