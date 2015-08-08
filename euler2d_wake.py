import matplotlib
matplotlib.use('agg')
matplotlib.interactive(False)

import sys
import time
from pylab import *
from numpy import *
from numpad import *

gamma, R = 1.4, 287.
DISS_COEFF = 0.01

def T_ratio(M2):
    return 1 + (gamma - 1) / 2 * M2

def p_ratio(M2):
    return T_ratio(M2)**(gamma / (gamma - 1))

T0, p0, M0 = 300., 101325., 0.05

rho0 = p0 / (R * T0)
c0 = sqrt(gamma * R * T0)
u0 = c0 * M0

R_plus_in = c0 * M0 + 2 * c0 / (gamma - 1)
S_in = p0 / rho0**gamma
R_minus_out = c0 * M0 - 2 * c0 / (gamma - 1)

Lx, Ly = 20., 5.
dx = dy = 0.1
dt = dx / u0
Nx, Ny = int(Lx / dx), int(Ly / dy)
x = arange(Nx) * dx + 0.5 * dx - 0.1 * Lx
y = arange(Ny) * dy + 0.5 * dy - 0.5 * Ly

x_ext = arange(Nx+2) * dx - 0.5 * dx - 0.1 * Lx
y_ext = arange(Ny+2) * dy - 0.5 * dy - 0.5 * Ly

i_obstacle = slice((x < -1).sum(), (x < 1).sum())
i_obstacle_ext = slice((x_ext < -1).sum(), (x_ext < 1).sum())
j_obstacle = slice((y < -.5).sum(), (y < .5).sum())
j_obstacle_ext = slice((y_ext < -.5).sum(), (y_ext < .5).sum())

def diffx(w):
    return (w[2:,1:-1] - w[:-2,1:-1]) / (2 * dx)

def diffy(w):
    return (w[1:-1,2:] - w[1:-1,:-2]) / (2 * dy)

def dissipation(r, u):
    # conservative, negative definite dissipation applied to r*d(ru)/dt
    def laplace(u):
        # d2udx2 = u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]
        # d2udy2 = u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]
        d2udx2 = vstack([u[1:2,:] - u[:1,:],
                         u[2:,:] - 2*u[1:-1,:] + u[:-2,:],
                         u[-1:,:] - u[-2:-1,:]])
        d2udy2 = hstack([u[:,1:2] - u[:,:1],
                         u[:,2:] - 2*u[:,1:-1] + u[:,:-2],
                         u[:,-1:] - u[:,-2:-1]])
        return (d2udx2 + d2udy2) / 4
    rho = r * r
    return laplace(rho * laplace(u))

def rhs(w):
    r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    u, v = ru / r, rv / r

    mass = diffx(r * ru) + diffy(r * rv)
    momentum_x = (diffx(ru*ru) + (r*ru)[1:-1,1:-1] * diffx(u)) / 2.0 \
               + (diffy(rv*ru) + (r*rv)[1:-1,1:-1] * diffy(u)) / 2.0 \
               + diffx(p)
    momentum_y = (diffx(ru*rv) + (r*ru)[1:-1,1:-1] * diffx(v)) / 2.0 \
               + (diffy(rv*rv) + (r*rv)[1:-1,1:-1] * diffy(v)) / 2.0 \
               + diffy(p)
    energy = gamma * (diffx(p * u) + diffy(p * v)) \
           - (gamma - 1) * (u[1:-1,1:-1] * diffx(p) + v[1:-1,1:-1] * diffy(p))

    dissipation_x = dissipation(r[1:-1,1:-1], u[1:-1,1:-1]) * c0 / dx
    dissipation_y = dissipation(r[1:-1,1:-1], v[1:-1,1:-1]) * c0 / dy
    dissipation_x *= DISS_COEFF
    dissipation_y *= DISS_COEFF
    momentum_x += dissipation_x
    momentum_y += dissipation_y
    energy -= u[1:-1,1:-1] * dissipation_x + v[1:-1,1:-1] * dissipation_y

    rhs_w = zeros(w[1:-1,1:-1].shape)
    rhs_w[:,:,0] = 0.5 * mass / r[1:-1,1:-1]
    rhs_w[:,:,1] = momentum_x / r[1:-1,1:-1]
    rhs_w[:,:,2] = momentum_y / r[1:-1,1:-1]
    rhs_w[:,:,-1] = energy

    rhs_w[i_obstacle,j_obstacle,1:3] += \
            100 * w[i_obstacle_ext, j_obstacle_ext, 1:3]
    return rhs_w

def apply_bc(w):
    w_ext = zeros([w.shape[0] + 2, w.shape[1] + 2, w.shape[-1]])
    w_ext[1:-1,1:-1] = w

    # inlet
    r, ru, rv, p = w[0,:,:].T
    c = sqrt(gamma) * sqrt(p) / r
    R_plus = R_plus_in
    R_minus = ru / r - 2 * c / (gamma - 1)
    S = S_in

    u_ext = (R_plus + R_minus) / 2
    c_ext = (R_plus - R_minus) * (gamma - 1) / 4
    r_ext = (c_ext**2 / gamma / S) ** (0.5 / (gamma - 1))
    p_ext = (r_ext * c_ext)**2 / gamma

    w_ext[0,1:-1,:] = transpose([r_ext, r_ext * u_ext, rv * 0, p_ext])

    # outlet
    r, ru, rv, p = w[-1,:,:].T
    c = sqrt(gamma) * sqrt(p) / r
    R_plus = ru / r + 2 * c / (gamma - 1)
    S = p / r**(2 * gamma)
    R_minus = R_minus_out

    u_ext = (R_plus + R_minus) / 2
    c_ext = (R_plus - R_minus) * (gamma - 1) / 4
    r_ext = (c_ext**2 / gamma / S) ** (0.5 / (gamma - 1))
    p_ext = (r_ext * c_ext)**2 / gamma

    w_ext[-1,1:-1,:] = transpose([r_ext, r_ext * u_ext, rv * 0, p_ext])

    # upper wall
    w_ext[:,0] = w_ext[:,1]
    w_ext[:,0,2] *= -1

    # lower wall
    w_ext[:,-1] = w_ext[:,-2]
    w_ext[:,-1,2] *= -1

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

wave = 1 + 0.0 * outer(sin(x / Lx * pi)**64, sin(y / Ly * pi)**32)
rho = rho0 * wave
p = p0 * wave**gamma
u = u0 * ones(wave.shape)
v = zeros(wave.shape)

w = zeros([Nx, Ny, 4])
w[:,:,0] = sqrt(rho)
w[:,:,1] = sqrt(rho) * u
w[:,:,2] = sqrt(rho) * v
w[:,:,-1] = p

w[:,:,1] *= 1 + 0.01 * (random.random(w[:,:,1].shape) - 0.5)

print(ddt_conserved(w, rhs(apply_bc(w))))

print(conserved(w))

figure(figsize=(18,10))
for iplot in range(5000):
    for istep in range(10):
        print(r'<br>')
        w = solve(midpoint_res, w, (w,), rel_tol=1E-9)
        # w = solve(midpoint_res, w, (w,), verbose=False)
        w.obliviate()
        sys.stdout.flush()
    print(r'<br>')
    print(conserved(w))
    r, ru, rv, p = w[:,:,0], w[:,:,1], w[:,:,2], w[:,:,-1]
    rho, u, v = r * r, ru / r, rv / r
    clf()
    u_range = linspace(-u0, u0 * 2, 100)
    subplot(2,1,1); contourf(x, y, value(u).T, u_range);
    axis('scaled'); colorbar()
    v_range = linspace(-u0, u0, 100)
    subplot(2,1,2); contourf(x, y, value(v).T, v_range);
    axis('scaled'); colorbar()
    savefig('fig{0:06d}.png'.format(iplot))

