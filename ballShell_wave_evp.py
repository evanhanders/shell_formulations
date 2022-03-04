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
nrS = nrB = int(nr/2)
base_resolution = (nφ, nθ, nr)
hi_resolution = (nφ, nθ, int(1.5*nr))
L_dealias = N_dealias = dealias = 1
dtype = np.complex128
Re  = float(args['--Re'])
Pr  = 1
nu = 1/Re
kappa = nu/Pr

Ri = 0.5
Ro = 1
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(Ri, Ro))

N2_mag = 1
N2_pow = 2

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
        basisB = d3.BallBasis(coords, resolution, radius=Ri, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
        basisS = d3.ShellBasis(coords, resolution, radii=(Ri, Ro), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
        S2_basisB = basisB.S2_basis()
        S2_basisS = basisS.S2_basis()
        φB, θB, rB = basisB.local_grids((1,1,1))
        φS, θS, rS = basisS.local_grids((1,1,1))

        #Ball fields
        uB = dist.VectorField(coords, name='uB', bases=basisB)
        pB = dist.Field(name='pB', bases=basisB)
        bB = dist.Field(name='bB', bases=basisB)
        tau_uB = dist.VectorField(coords, name='tau_uB', bases=S2_basisB)
        tau_bB = dist.Field(name='tau_bB', bases=S2_basisB)

        # Shell Fields
        uS = dist.VectorField(coords, name='uS', bases=basisS)
        pS = dist.Field(name='pS', bases=basisS)
        bS = dist.Field(name='bS', bases=basisS)
        tau_u1S = dist.VectorField(coords, name='tau_u1S', bases=S2_basisS)
        tau_u2S = dist.VectorField(coords, name='tau_u2S', bases=S2_basisS)
        tau_b1S = dist.Field(name='tau_b1S', bases=S2_basisS)
        tau_b2S = dist.Field(name='tau_b2S', bases=S2_basisS)

        # Constants
        omega = dist.Field(name='omega')
        tau_p = dist.Field(name='tau_p')

        # NCCs
        grad_b0B = dist.VectorField(coords, name='grad_b0B', bases=basisB.radial_basis)
        rvecB    = dist.VectorField(coords, name='rvecB', bases=basisB.radial_basis)
        erB      = dist.VectorField(coords, name='erB', bases=basisB.radial_basis)
        grad_b0S = dist.VectorField(coords, name='grad_b0S', bases=basisS.radial_basis)
        rvecS    = dist.VectorField(coords, name='rvecS', bases=basisS.radial_basis)
        erS      = dist.VectorField(coords, name='erS', bases=basisS.radial_basis)

        grad_b0B['g'][2] = N2_mag*(1 + rB**N2_pow)
        rvecB['g'][2] = rB
        erB['g'][2] = 1
        grad_b0S['g'][2] = N2_mag*(1 + rS**N2_pow)
        rvecS['g'][2] = rS
        erS['g'][2] = 1


        logger.info('using formulation {}'.format(formulation_index))
        # Lift operators for boundary conditions
        lift_basis_k1S = basisS.clone_with(k=1)
        lift_basis_k2S = basisS.clone_with(k=2)
        lift_basis_B = basisB.clone_with(k=0)
        lift_k1S   = lambda A, n: d3.Lift(A, lift_basis_k1S, n)
        lift_k2S   = lambda A, n: d3.Lift(A, lift_basis_k2S, n)
        liftB      = lambda A: d3.Lift(A, lift_basis_B, -1)

        BC_uB = liftB(tau_uB)
        BC_bB = liftB(tau_bB)
        grad_uB = d3.grad(uB)
        grad_bB = d3.grad(bB)
        div_uB = d3.div(uB) + d3.dot(erB, liftB(tau_uB))
        if formulation_index in (0, 3):
            logger.info('using standard FOF')
            BC_uS = lift_k1S(tau_u1S, -1)
            BC_bS = lift_k1S(tau_b1S, -1)
            grad_uS = d3.grad(uS) + erS*lift_k1S(tau_u2S, -1)
            grad_bS = d3.grad(bS) + erS*lift_k1S(tau_b2S, -1)
            div_uS = d3.trace(grad_uS)
        elif formulation_index in (1, 4):
            logger.info('using k=2 formulation')
            BC_uS = lift_k2S(tau_u1S, -1) + lift_k2S(tau_u2S, -2)
            BC_bS = lift_k2S(tau_b1S, -1) + lift_k2S(tau_b2S, -2)
            grad_bS = d3.grad(bS)
            grad_uS = d3.grad(uS)
            div_uS = d3.div(uS) + d3.dot(erS, lift_k2S(tau_u1S, -1))
        elif formulation_index in (2, 5):
            logger.info('using k=2 with k=1 blend')
            BC_uS = lift_k1S(tau_u1S, -1) + lift_k2S(tau_u2S, -2)
            BC_bS = lift_k1S(tau_b1S, -1) + lift_k2S(tau_b2S, -2)
            grad_bS = d3.grad(bS)
            grad_uS = d3.grad(uS)
            div_uS = d3.div(uS) + d3.dot(rvecS, lift_k1S(tau_u2S, -1))
    
        EB = 0.5*(grad_uB + d3.transpose(grad_uB))
        ES = 0.5*(grad_uS + d3.transpose(grad_uS))

        ddt = lambda A: -1j*omega*A

        if formulation_index in (3, 4):
            logger.info("conditioning out ell = 0")
            problem = d3.EVP([ bB, pB, uB, bS, pS, uS, tau_bB, tau_uB, tau_b1S, tau_b2S, tau_u1S, tau_u2S], eigenvalue=omega, namespace=locals())

            problem.add_equation("ddt(bB) + dot(uB, grad_b0B) - kappa*div(grad_bB) + BC_bB = 0")
            problem.add_equation("div_uB = 0", condition="nθ != 0")
            problem.add_equation("ddt(uB) - bB*rvecB + grad(pB) - nu*div(grad_uB) + BC_uB = 0", condition="nθ != 0")
            problem.add_equation("pB = 0", condition="nθ == 0")
            problem.add_equation("uB = 0", condition="nθ == 0")
            problem.add_equation("ddt(bS) + dot(uS, grad_b0S) - kappa*div(grad_bS) + BC_bS = 0")
            problem.add_equation("div_uS = 0", condition="nθ != 0")
            problem.add_equation("ddt(uS) - bS*rvecS + grad(pS) - nu*div(grad_uS) + BC_uS = 0", condition="nθ != 0")
            problem.add_equation("pS = 0", condition="nθ == 0")
            problem.add_equation("uS = 0", condition="nθ == 0")


            problem.add_equation("uS(r=Ri) - uB(r=Ri) = 0", condition="nθ != 0")
            problem.add_equation("pS(r=Ri) - pB(r=Ri) = 0", condition="nθ != 0")
            problem.add_equation("angular(radial(EB(r=Ri) - ES(r=Ri))) = 0", condition="nθ != 0")

            problem.add_equation("radial(uS(r=Ro)) = 0", condition="nθ != 0")
            problem.add_equation("angular(radial(ES(r=Ro))) = 0", condition="nθ != 0")
            problem.add_equation("tau_uB = 0", condition="nθ == 0")
            problem.add_equation("tau_u1S = 0", condition="nθ == 0")
            problem.add_equation("tau_u2S = 0", condition="nθ == 0")


            problem.add_equation("bS(r=Ri) - bB(r=Ri) = 0")
            problem.add_equation("radial(grad_bS(r=Ri) - grad_bB(r=Ri)) = 0")
            problem.add_equation("radial(grad_bS(r=Ro)) = 0")
        else:
            logger.info("Using tau_p")
            problem = d3.EVP([ bB, pB, uB, bS, pS, uS, tau_p, tau_bB, tau_uB, tau_b1S, tau_b2S, tau_u1S, tau_u2S], eigenvalue=omega, namespace=locals())

            problem.add_equation("ddt(bB) + dot(uB, grad_b0B) - kappa*div(grad_bB) + BC_bB = 0")
            problem.add_equation("div_uB + tau_p = 0")
            problem.add_equation("ddt(uB) - bB*rvecB + grad(pB) - nu*div(grad_uB) + BC_uB = 0")
            problem.add_equation("ddt(bS) + dot(uS, grad_b0S) - kappa*div(grad_bS) + BC_bS = 0")
            problem.add_equation("div_uS + tau_p= 0")
            problem.add_equation("ddt(uS) - bS*rvecS + grad(pS) - nu*div(grad_uS) + BC_uS = 0")

            problem.add_equation("integ(pB) + integ(pS) = 0")

            problem.add_equation("uS(r=Ri) - uB(r=Ri) = 0")
            problem.add_equation("pS(r=Ri) - pB(r=Ri) = 0")
            problem.add_equation("angular(radial(EB(r=Ri) - ES(r=Ri))) = 0")
            problem.add_equation("radial(uS(r=Ro)) = 0")
            problem.add_equation("angular(radial(ES(r=Ro))) = 0")

            problem.add_equation("bS(r=Ri) - bB(r=Ri) = 0")
            problem.add_equation("radial(grad_bS(r=Ri) - grad_bB(r=Ri)) = 0")
            problem.add_equation("radial(grad_bS(r=Ro)) = 0")



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
