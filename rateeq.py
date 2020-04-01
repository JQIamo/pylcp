#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for solving the rate equations.
"""
import numpy as np
import copy
from scipy.optimize import minimize, fsolve
from scipy.integrate import solve_ivp
from inspect import signature
from .lasers import laserBeams

#@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

class force_profile():
    def __init__(self, R, V, laserBeams, hamiltonian,
                 contents=['Rijl','Neq','f','F']):
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        if not isinstance(V, np.ndarray):
            V = np.array(V)

        if R.shape[0] != 3 or V.shape[0] != 3:
            raise TypeError('R and V must have first dimension of 3.')

        self.R = copy.copy(R)
        self.V = copy.copy(V)

        self.Rijl = {}
        self.f = {}
        for key in laserBeams:
             self.Rijl[key] = np.zeros(
                 R[0].shape+(len(laserBeams[key].beam_vector),
                             hamiltonian.blocks[hamiltonian.laser_keys[key]].n,
                             hamiltonian.blocks[hamiltonian.laser_keys[key]].m)
                 )
             self.f[key] = np.zeros(R.shape + (len(laserBeams[key].beam_vector),))

        self.Neq = np.zeros(R[0].shape + (hamiltonian.n,))
        self.F = np.zeros(R.shape)

    def store_data(self, ind, Rijl, Neq, f, F):
        self.Neq[ind] = Neq

        for key in Rijl:
            self.Rijl[key][ind] = Rijl[key]

        for key in f:
            for jj in range(3):
                self.f[key][(jj,) + ind] = f[key][jj]

        for jj in range(3):
            self.F[(jj,) + ind] = F[jj]


class rateeq():
    """
    The class rateeq prduces a set of rate equations for a given
    position and velocity and provides methods for solving them appropriately.
    """
    def __init__(self, lasers, magField, hamiltonian,
                 r=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, 0.0]),
                 svd_eps=1e-10):
        """
        First step is to save the imported laserBeams, magField, and
        hamiltonian.
        """
        self.hamiltonian = copy.copy(hamiltonian)

        self.laserBeams = {} # Laser beams are meant to be dictionary,
        if isinstance(lasers, list):
            self.laserBeams['g->e'] = copy.copy(laserBeams(lasers)) # Assume label is g->e
        elif isinstance(lasers, laserBeams):
            self.laserBeams['g->e'] = copy.copy(lasers) # Again, assume label is g->e
        elif isinstance(lasers, dict):
            self.laserBeams = copy.copy(lasers) # Now, assume that everything is the same.
        else:
            raise ValueError('laserBeams is not a valid type.')

        # Check that laser beam keys and Hamiltonian keys match.
        for laser_key in self.laserBeams.keys():
            if not laser_key in self.hamiltonian.laser_keys.keys():
                raise ValueError('laserBeams dictionary keys %s does not have '+
                                 'a corresponding key the Hamiltonian d_q.' %
                                 laser_key)

        self.magField = copy.copy(magField)

        self.svd_eps = svd_eps

        # Check function signatures for any time dependence:
        self.tdepend = {}
        self.tdepend['B'] = False
        """self.tdepend['pol'] = False
        self.tdepend['kvec'] = False
        self.tdepend['det'] = False
        self.tdepend['beta'] = False"""

        if 't' in str(signature(self.magField)):
            self.tdepend['B'] = True
        """for key in self.laserBeams:
            for beam in self.laserBeams[key]:
                if not beam.pol_sig is None and 't' in beam.pol_sig:
                    self.tdepend['pol'] = True
                if not beam.kvec_sig is None and 't' in beam.kvec_sig:
                    self.tdepend['kvec'] = True
                if not beam.delta_sig is None and 't' in beam.delta_sig:
                    self.tdepend['det'] = True
                if not beam.beta_sig is None and 't' in beam.beta_sig:
                    self.tdepend['beta'] = True"""

        # Reset the current solution to None
        self.set_initial_position_and_velocity(r, v)

        # Set up a dictionary to store the profiles.
        self.profile = {}

    def construct_evolution_matrix_decay(self, rotated_ham):
        Rev_decay = np.zeros((self.hamiltonian.n, self.hamiltonian.n))

        # Go through each of the blocks and calculate the decay rate
        # contributions to the rate equations.  This would be simpler with
        # rotated_ham.make_full_matrices(), but I would imagine it would
        # be significantly slower.
        for ll, block in enumerate(np.diagonal(rotated_ham.blocks)):
            n = sum(rotated_ham.ns[:ll])

            # Calculate the decays first, provided we are not in the lowest
            # manifold:
            if ll>0:
                for mm, other_block in enumerate(rotated_ham.blocks[:ll, ll]):
                    if not other_block is None:
                        for jj in range(block.n):
                            Rev_decay[n+jj, n+jj] -= np.sum(np.sum(abs2(
                                other_block.matrix[:, :, jj]
                                )))

            # Calculate the decays into of the manifold, provided we are not
            # in the most excited one:
            if ll<len(rotated_ham.ns)-1:
                for mm, other_block in enumerate(rotated_ham.blocks[ll, ll+1:]):
                    if not other_block is None:
                        m = sum(rotated_ham.ns[:ll+1+mm])
                        Rev_decay[n:n+block.n, m:m+other_block.m] += \
                        np.sum(abs2(other_block.matrix), axis=0)

        return Rev_decay


    def construct_evolution_matrix(self, r, v, t=0.):
        """
        Method to generate the rate equation evolution matrix.  Expects
        several arguments:
            r: a three-element vector specifying the position of interest
            v: a three-element vector specifying the velocity of interest
        """
        # What is the magnetic field?
        if self.tdepend['B']:
            B = self.magField(r, t)
        else:
            B = self.magField(r)

        # Calculate its magnitude:
        Bmag = np.linalg.norm(B, axis=0)

        # Calculate the Bhat direction:
        if np.abs(Bmag) > 0:
            Bhat = B/Bmag
        else:
            Bhat = np.zeros(B.shape)
            Bhat[-1] = 1.0

        # Diagonalize at Bmag.  This is required if the Hamiltonian has any
        # non-diagonal elements of the Hamiltonian that are dependent on Bz.
        # This occurs, for example, with atoms in the large
        rotated_ham, already_diagonal = self.hamiltonian.diag_static_field(Bmag)
        if not already_diagonal:
            # Reconstruct the decay matrix to
            self.Rev_decay = self.construct_evolution_matrix_decay(rotated_ham)
        else:
            if not hasattr(self, 'Rev_decay'):
                self.Rev_decay = self.construct_evolution_matrix_decay(rotated_ham)

        # Now add in the lasers
        self.Rev = np.zeros((self.hamiltonian.n, self.hamiltonian.n))
        self.Rev += self.Rev_decay
        self.Rijl = {}

        for key in self.laserBeams:
            # Extract the relevant dijq matrix:
            ind = rotated_ham.laser_keys[key]
            dijq = rotated_ham.blocks[ind].matrix

            # Extract the energies:
            E1 = np.diag(rotated_ham.blocks[ind[0],ind[0]].matrix)
            E2 = np.diag(rotated_ham.blocks[ind[1],ind[1]].matrix)

            # Initialize the pumping matrix:
            self.Rijl[key] = np.zeros((len(self.laserBeams[key].beam_vector),) +
                                      dijq.shape[1:])

            # Grab the laser parameters:
            (kvecs, betas, pols, deltas) = self.laserBeams[key].return_parameters(r, t)
            projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

            # Loop through each laser beam driving this transition:
            for ll, (kvec, beta, proj, delta) in enumerate(zip(kvecs, betas, projs, deltas)):
                for ii in range(dijq.shape[1]):
                    for jj in range(dijq.shape[2]):
                        fijq = np.abs(np.dot(dijq[:, ii, jj], proj))**2
                        if fijq > 0:
                            # Finally, calculate the scattering rate the polarization
                            # onto the appropriate basis:
                            """Rijl[ll, ii, jj] = beam.beta(r)/2*fijq/\
                            (1 + 4*(beam.delta - (H0[ng+jj, ng+jj] - H0[ii, ii]) -
                                    np.dot(kvec, v))**2)"""
                            self.Rijl[key][ll, ii, jj] = beta/2*\
                                fijq/(1 + 4*((E2[jj] - E1[ii]) + delta -
                                             np.dot(kvec, v))**2)

            # Now add the pumping rates into the rate equation propogation matrix:
            n = sum(rotated_ham.ns[:ind[0]])
            m = sum(rotated_ham.ns[:ind[1]])
            for ii in range(self.Rijl[key].shape[1]):
                for jj in range(self.Rijl[key].shape[2]):
                    self.Rev[n+ii, n+ii] += -np.sum(self.Rijl[key][:, ii, jj])
                    self.Rev[n+ii, m+jj] += np.sum(self.Rijl[key][:, ii, jj])
                    self.Rev[m+jj, n+ii] += np.sum(self.Rijl[key][:, ii, jj])
                    self.Rev[m+jj, m+jj] += -np.sum(self.Rijl[key][:, ii, jj])

        return self.Rev, self.Rijl


    def equilibrium_populations(self, r, v, t, return_details=False):
        """
        Returns the equilibrium values of the rate equation matrix Rev by
        singular matrix decomposition.
        """
        Rev, Rijl = self.construct_evolution_matrix(r, v, t)

        # Find the singular values:
        U, S, VH = np.linalg.svd(Rev)

        Neq = np.compress(S <= self.svd_eps, VH, axis=0).T
        Neq /= np.sum(Neq)

        if Neq.shape[1] > 1:
            Neq = np.nan*Neq[:, 0]
            #raise ValueError("more than one equilbrium state found")

        # The operations above return a column vector, this collapses the column
        # vector into an array:
        Neq = Neq.flatten()

        if return_details:
            return (Neq, Rev, Rijl)
        else:
            return Neq


    def force(self, r, t, Rijl, Npop):
        F = np.zeros((3,))
        f = {}

        for key in self.laserBeams:
            f[key] = np.zeros((3, len(self.laserBeams[key].beam_vector)))

            ind = self.hamiltonian.laser_keys[key]
            n = sum(self.hamiltonian.ns[:ind[0]])
            m = sum(self.hamiltonian.ns[:ind[1]])
            for ll, beam in enumerate(self.laserBeams[key].beam_vector):
                # If kvec is callable, evaluate kvec:
                kvec = beam.return_kvec(r, t)

                for ii in range(Rijl[key].shape[1]):
                    for jj in range(Rijl[key].shape[2]):
                        if Rijl[key][ll, ii, jj]>0:
                            f[key][:, ll] += kvec*Rijl[key][ll, ii, jj]*\
                                (Npop[n+ii] - Npop[m+jj])

            F += np.sum(f[key], axis=1)

        return F, f

    def set_initial_position_and_velocity(self, r0, v0):
        self.set_initial_position(r0)
        self.set_initial_velocity(v0)

    def set_initial_position(self, r0):
        self.r0 = r0

    def set_initial_velocity(self, v0):
        self.v0 = v0

    def set_initial_pop(self, N0):
        self.N0 = N0

    def set_initial_pop_from_equilibrium(self):
        self.N0 = self.equilibrium_populations(self.r0, self.v0, t=0)


    def evolve_populations(self, t_span, **kwargs):
        """
        This function evolves the populations only.  It assumes that the atom
        has velocity v, but is yet moving through space.  This allows us to
        determine equilibrium populations and forces on atoms at every point
        in space.
        """
        if any([self.tdepend[key] for key in self.tdepend.keys()]):
            raise NotImplementedError('Time dependence not yet implemented.')
        else:
            Rev, Rijl = self.construct_evolution_matrix(self.r0, self.v0)
            self.sol = solve_ivp(lambda t, N: Rev @ N, t_span, self.N0, **kwargs)


    def evolve_motion(self, t_span, **kwargs):
        """
        This method evolves the rate equations and atomic motion while in both
        changing laser fields and magnetic fields.
        """
        hold_axis = kwargs.pop('freeze_axis', [False, False, False])
        mass_ratio = kwargs.pop('mass_ratio', 1e-3)
        def motion(t, y):
            N = y[:-6]
            v = y[-6:-3]
            r = y[-3:]

            Rev, Rijl = self.construct_evolution_matrix(r, v, t)
            F, f_laser = self.force(Rijl, N)
            F[hold_axis] = 0.

            dydt = np.concatenate((Rev @ N, mass_ratio*F, v))
            if np.any(np.isnan(dydt)):
                raise ValueError('Enountered a NaN!')

            return dydt

        y0 = np.concatenate((self.N0, self.v0, self.r0))
        self.sol = solve_ivp(motion, t_span, y0, **kwargs)

    def find_equilibrium_force(self, **kwargs):
        """
        Determines the force from the lasers at position r based on the
        populations Npop, scattering rate Rijl, and laserBeams.
        """
        return_details = kwargs.pop('return_details', False)

        if any([self.tdepend[key] for key in self.tdepend.keys()]):
            raise NotImplementedError('Time dependence not yet implemented.')
        else:
            N_eq, Rev, Rijl = self.equilibrium_populations(
                self.r0, self.v0, t=0, return_details=True
                )
            F_eq, f_eq = self.force(self.r0, 0., Rijl, N_eq)

        if return_details:
            return F_eq, f_eq, N_eq, Rijl
        else:
            return F_eq

    def generate_force_profile(self, R, V, name=None):
        """
        Method that maps out the equilbirium pupming rate, populations,
        and forces:
        """
        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams,
                                           self.hamiltonian)

        it = np.nditer([R[0], R[1], R[2], V[0], V[1], V[2]],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['readonly'],
                                 ['readonly'], ['readonly'], ['readonly']])

        for (x, y, z, vx, vy, vz) in it:
            # Construct the rate equations:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            # Update position (use multi_index), populations, and forces.
            self.set_initial_position_and_velocity(r, v)
            try:
                F, f, Neq, Rijl = self.find_equilibrium_force(
                    return_details=True
                    )
            except:
                raise ValueError(
                    "Unable to find solution at " +\
                    "r=({0:0.2f},{1:0.2f},{2:0.2f})".format(x, y, z) + " and " +
                    "v=({0:0.2f},{1:0.2f},{2:0.2f})".format(vx, vy, vz)
                    )

            self.profile[name].store_data(it.multi_index, Rijl, Neq, f, F)


class trap(rateeq):
    def __init__(self, laserBeams, magField, hamiltonian, svd_eps=1e-10):
        super(trap, self).__init__(laserBeams, magField, hamiltonian,
                                   svd_eps=svd_eps)
        self.r0 = np.zeros((3,))


    def find_equilibrium_position(self, axes=[2], zlim=5., Npts=51,
                                  initial_search=True):
        # Next, find the equilibrium point in z, and evaluate derivatives there:
        r0i = np.zeros((3,))
        z = np.linspace(-zlim, zlim, Npts)

        if initial_search:
            for axis in axes:
                v = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r[axis] = z

                self.generate_force_profile(r, v, name='root_search')

                z_possible = z[np.where(np.diff(np.sign(
                    self.profile['root_search'].F[axis]))<0)[0]]

                if z_possible.size>0:
                    if z_possible.size>1:
                        ind = np.argmin(z_possible**2)
                    else:
                        ind = 0
                    r0i[axis] = z_possible[ind]
                else:
                    r0i[axis] = np.nan

                del self.profile['root_search']

        if len(axes)>1:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return np.sum(F**2)

            if np.sum(np.isnan(r0i)) == 0:
                # Find the center of the trap:
                result = minimize(simple_wrapper, r0i[axes], method='SLSQP')
                if result.success:
                    self.r0[axes] = result.x
                else:
                    self.r0[axes] = np.nan
            else:
                self.r0 = np.nan
        else:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return F[axes]

            if np.sum(np.isnan(r0i)) == 0:
                self.r0[axes] = fsolve(simple_wrapper, r0i[axes])[0]
            else:
                self.r0[axes] = np.nan

        return self.r0

    def trapping_frequencies(self, axes=[0, 2], r=None, eps=0.01):
        self.omega = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None:
            r = self.r0

        for axis in axes:
            if not np.isnan(r[axis]):
                rpmdri = np.tile(r, (2,1)).T
                rpmdri[axis, 1] += eps[axis]
                rpmdri[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(rpmdri[:, jj],
                                                      np.zeros((3,1)))
                    f = self.find_equilibrium_force()

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.omega[axis] = np.sqrt(-np.diff(F)/(2*eps[axis]))
                else:
                    self.omega[axis] = 0
            else:
                self.omega[axis] = 0

        return self.omega[axes]

    def damping_coeff(self, axes=[0, 2], r=None, eps=0.01):
        self.beta = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None:
            r = self.r0

        for axis in axes:
            if not np.isnan(r[axis]):
                vpmdvi = np.zeros((3,2))
                vpmdvi[axis, 1] += eps[axis]
                vpmdvi[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(r, vpmdvi[:, jj])
                    f = self.find_equilibrium_force()

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.beta[axis] = -np.diff(F)/(2*eps[axis])
                else:
                    self.beta[axis] = 0
            else:
                self.beta[axis] = 0

        return self.beta[axes]
