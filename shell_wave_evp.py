"""
d3 EVP script to see if different tau formulations lead to linear instabilities in a problem that should be stable.

Usage:
    shell_wave_evp.py [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 3]
    --nr=<res>          Number of radial grid points in ball (Nmax+1)   [default: 16]
"""
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Parameters
args   = docopt(__doc__)
nθ  = int(args['--ntheta'])
nφ  = int(2*nθ)
nr = int(args['--nr'])
base_resolution = (nφ, nθ, nr)
hi_resolution = (nφ, nθ, int(1.5*nr))
L_dealias = N_dealias = dealias = 1
dtype = np.complex128
Re  = float(args['--Re'])
Pr  = 1
nu = 1/Re
kappa = nu/Pr

Ri = 1.1
Ro = 1.5
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(Ri, Ro))

N2_mag = 0
N2_pow = 1

run_formulations = [0, 1, 2, 3, 4, 5]
for formulation_index in run_formulations:
    print('\n\n')
    growths = dict()
    for ell in range(nθ):
        growths[ell] = []
    for resolution in base_resolution, hi_resolution:
        logger.info("Solving with Resolution: {}".format(resolution))
        # Bases
        coords  = d3.SphericalCoordinates('φ', 'θ', 'r')
        dist    = d3.Distributor((coords,), dtype=dtype)
        basis = d3.ShellBasis(coords, resolution, radii=(Ri, Ro), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
        S2_basis = basis.S2_basis()
        φ, θ, r = basis.local_grids((1,1,1))

        # Fields
        u = dist.VectorField(coords, name='u', bases=basis)
        p = dist.Field(name='p', bases=basis)
        b = dist.Field(name='b', bases=basis)
        omega = dist.Field(name='omega')
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=S2_basis)
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=S2_basis)
        tau_b1 = dist.Field(name='tau_b1', bases=S2_basis)
        tau_b2 = dist.Field(name='tau_b2', bases=S2_basis)
        tau_p = dist.Field(name='tau_p')

        # NCCs
        grad_b0 = dist.VectorField(coords, name='grad_b0', bases=basis.radial_basis)
        rvec    = dist.VectorField(coords, name='rvec', bases=basis.radial_basis)
        er      = dist.VectorField(coords, name='er', bases=basis.radial_basis)

        grad_b0['g'][2] = N2_mag*(1 + ((r-Ri)/(Ro-Ri))**N2_pow)
        rvec['g'][2] = r
        er['g'][2] = 1


        logger.info('using formulation {}'.format(formulation_index))
        # Lift operators for boundary conditions
        lift_basis_k1 = basis.clone_with(k=1)
        lift_basis_k2 = basis.clone_with(k=2)
        lift_k1   = lambda A, n: d3.Lift(A, lift_basis_k1, n)
        lift_k2   = lambda A, n: d3.Lift(A, lift_basis_k2, n)
        if formulation_index in (0, 3):
            logger.info('using standard FOF')
            BC_u = lift_k1(tau_u1, -1)
            BC_b = lift_k1(tau_b1, -1)
            grad_u = d3.grad(u) + er*lift_k1(tau_u2, -1)
            grad_b = d3.grad(b) + er*lift_k1(tau_b2, -1)
            div_u = d3.trace(grad_u)
        elif formulation_index in (1, 4):
            logger.info('using k=2 formulation')
            BC_u = lift_k2(tau_u1, -1) + lift_k2(tau_u2, -2)
            BC_b = lift_k2(tau_b1, -1) + lift_k2(tau_b2, -2)
            grad_b = d3.grad(b)
            grad_u = d3.grad(u)
            div_u = d3.div(u) + d3.dot(er, lift_k2(tau_u1, -1))
        elif formulation_index in (2, 5):
            logger.info('using k=2 with k=1 blend')
            BC_u = lift_k1(tau_u1, -1) + lift_k2(tau_u2, -2)
            BC_b = lift_k1(tau_b1, -1) + lift_k2(tau_b2, -2)
            grad_b = d3.grad(b)
            grad_u = d3.grad(u)
            div_u = d3.div(u) + d3.dot(rvec, lift_k1(tau_u2, -1))
    
        E = 0.5*(grad_u + d3.transpose(grad_u))


        ddt = lambda A: -1j*omega*A

        if formulation_index in (3, 4, 5):
            logger.info("conditioning out ell = 0")
            problem = d3.EVP([ b, p, u, tau_b1, tau_b2, tau_u1, tau_u2], eigenvalue=omega, namespace=locals())

            problem.add_equation("ddt(b) + dot(u, grad_b0) - kappa*div(grad_b) + BC_b = 0")
            problem.add_equation("div_u = 0", condition="nθ != 0")
            problem.add_equation("ddt(u) - b*er + grad(p) - nu*div(grad_u) + BC_u = 0", condition="nθ != 0")
            problem.add_equation("p = 0", condition="nθ == 0")
            problem.add_equation("u = 0", condition="nθ == 0")

            problem.add_equation("radial(u(r=Ro)) = 0", condition="nθ != 0")
            problem.add_equation("radial(u(r=Ri)) = 0", condition="nθ != 0")
            problem.add_equation("angular(radial(E(r=Ro))) = 0", condition="nθ != 0")
            problem.add_equation("angular(radial(E(r=Ri))) = 0", condition="nθ != 0")
            problem.add_equation("tau_u1 = 0", condition="nθ == 0")
            problem.add_equation("tau_u2 = 0", condition="nθ == 0")
            problem.add_equation("radial(grad_b(r=Ro)) = 0")
            problem.add_equation("radial(grad_b(r=Ri)) = 0")
        else:
            logger.info("Using tau_p")
            problem = d3.EVP([ b, p, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], eigenvalue=omega, namespace=locals())

            problem.add_equation("ddt(b) + dot(u, grad_b0) - kappa*div(grad_b) + BC_b = 0")
            problem.add_equation("div_u + tau_p = 0")
            problem.add_equation("ddt(u) - b*er + grad(p) - nu*div(grad_u) + BC_u = 0")

            problem.add_equation("radial(u(r=Ro)) = 0")
            problem.add_equation("radial(u(r=Ri)) = 0")
            problem.add_equation("angular(radial(E(r=Ro))) = 0")
            problem.add_equation("angular(radial(E(r=Ri))) = 0")
            problem.add_equation("radial(grad_b(r=Ro)) = 0")
            problem.add_equation("radial(grad_b(r=Ri)) = 0")
            problem.add_equation("integ(p) = 0")

        logger.info("Problem built")
        # Solver
        solver = problem.build_solver()
        logger.info("solver built")

        if dist.comm_cart.size == 1:
            import matplotlib.pyplot as plt 
            figure = plt.figure(figsize=(8,4))
            for subproblem in solver.subproblems:
                ell = subproblem.group[1]
                sp = subproblem
                LHS = sp.pre_left.T @ (sp.M_min + 0.5*sp.L_min)
                plt.imshow(np.log10(np.abs(LHS.A)))
                plt.colorbar()
                plt.savefig("matrices/ell_%03i.png" %ell, dpi=600)
                plt.clf()
                cond = np.linalg.cond((sp.M_min + 0.5*sp.L_min).A)
                print('subproblem group {}, condition: {:.4e}'.format(subproblem.group, cond))

        for subproblem in solver.subproblems:
            ell = subproblem.group[1]
            logger.info('solving {} / ell {}'.format(resolution, ell))

            solver.solve_dense(subproblem)

            values = solver.eigenvalues
            vectors = solver.eigenvectors

            cond1 = np.isfinite(values)
            values = values[cond1]
            vectors = vectors[cond1]

            growth = values.imag
            growths[ell].append(growth)

    cutoff = 1e-2
    for ell, growthvals in growths.items():
        good_growths = []
        for i, v1 in enumerate(growthvals[0]):
            for j, v2 in enumerate(growthvals[1]):
                goodness = np.abs(v1 - v2)/np.abs(v1)
                if goodness < cutoff:
                    good_growths.append(v1)
                    break

        logger.info('solve_dense ell = {}'.format(ell) + ', max growth: {:.2e}, num > 0: {}'.format(np.max(good_growths), np.sum(np.array(good_growths) > 0)))
