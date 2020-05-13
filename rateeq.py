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
from .fields import laserBeams, magField
from .common import random_vector, base_force_profile, progressBar
from .integration_tools import solve_ivp_random
from scipy.interpolate import interp1d

#@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

class force_profile(base_force_profile):
    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        # Add in the specific instance of the Rijl:
        self.Rijl = {}
        for key in laserBeams:
            self.Rijl[key] = np.zeros(
                self.R[0].shape+(len(laserBeams[key].beam_vector),
                hamiltonian.blocks[hamiltonian.laser_keys[key]].n,
                hamiltonian.blocks[hamiltonian.laser_keys[key]].m)
                )

    def store_data(self, ind, Neq, F, F_laser, F_mag, Rijl):
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for key in Rijl:
            self.Rijl[key][ind] = Rijl[key]


class rateeq(object):
    """
    The class rateeq prduces a set of rate equations for a given
    position and velocity and provides methods for solving them appropriately.
    """
    def __init__(self, *args, **kwargs):
        """
        First step is to save the imported laserBeams, magField, and
        hamiltonian.
        """
        if len(args) < 3:
            raise ValueError('You must specify laserBeams, magField, and Hamiltonian')
        elif len(args) == 3:
            self.constant_accel = np.array([0., 0., 0.])
        elif len(args) == 4:
            if not isinstance(args[3], np.ndarray):
                raise TypeError('Constant acceleration must be an numpy array.')
            elif args[3].size != 3:
                raise ValueError('Constant acceleration must have length 3.')
            else:
                self.constant_accel = args[3]
        else:
            raise ValueError('No more than four positional arguments accepted.')

        # Add the Hamiltonian:
        self.hamiltonian = copy.copy(args[2])
        self.hamiltonian.make_full_matrices()

        # Add lasers:
        self.laserBeams = {} # Laser beams are meant to be dictionary,
        if isinstance(args[0], list):
            self.laserBeams['g->e'] = copy.copy(laserBeams(args[0])) # Assume label is g->e
        elif isinstance(args[0], laserBeams):
            self.laserBeams['g->e'] = copy.copy(args[0]) # Again, assume label is g->e
        elif isinstance(args[0], dict):
            for key in args[0].keys():
                if not isinstance(args[0][key], laserBeams):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(args[0]) # Now, assume that everything is the same.
        else:
            raise TypeError('laserBeams is not a valid type.')

        # Add in magnetic field:
        if callable(args[1]) or isinstance(args[1], np.ndarray):
            self.magField = magField(args[1])
        elif isinstance(args[1], magField):
            self.magField = copy.copy(args[1])
        else:
            raise TypeError('The magnetic field must be either a lambda ' +
                            'function or a magField object.')

        # Now handle keyword arguments:
        r=kwargs.pop('r', np.array([0., 0., 0.]))
        v=kwargs.pop('v', np.array([0., 0., 0.]))

        self.include_mag_forces = kwargs.pop('include_mag_forces', True)
        self.svd_eps = kwargs.pop('svd_eps', 1e-10)

        # Check function signatures for any time dependence:
        self.tdepend = {}
        self.tdepend['B'] = False
        """self.tdepend['pol'] = False
        self.tdepend['kvec'] = False
        self.tdepend['det'] = False
        self.tdepend['beta'] = False"""

        """if 't' in str(signature(self.magField)):
            self.tdepend['B'] = True
        for key in self.laserBeams:
            for beam in self.laserBeams[key]:
                if not beam.pol_sig is None and 't' in beam.pol_sig:
                    self.tdepend['pol'] = True
                if not beam.kvec_sig is None and 't' in beam.kvec_sig:
                    self.tdepend['kvec'] = True
                if not beam.delta_sig is None and 't' in beam.delta_sig:
                    self.tdepend['det'] = True
                if not beam.beta_sig is None and 't' in beam.beta_sig:
                    self.tdepend['beta'] = True"""

        # If the matrix is diagonal, we get to do something cheeky.  Let's just
        # construct the decay part of the evolution once:
        if np.all(self.hamiltonian.diagonal):
            self.construct_evolution_matrix_decay(self.hamiltonian)

        # Reset the current solution to
        self.set_initial_position_and_velocity(r, v)

        # Set up a dictionary to store the profiles.
        self.profile = {}

    def construct_evolution_matrix_decay(self, rotated_ham):
        self.Rev_decay = np.zeros((self.hamiltonian.n, self.hamiltonian.n))

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
                        for jj in range(rotated_ham.ns[ll]):
                            self.Rev_decay[n+jj, n+jj] -= np.sum(np.sum(abs2(
                                other_block.matrix[:, :, jj]
                                )))

            # Calculate the decays into of the manifold, provided we are not
            # in the most excited one:
            if ll<len(rotated_ham.ns)-1:
                for mm, other_block in enumerate(rotated_ham.blocks[ll, ll+1:]):
                    if not other_block is None:
                        m = sum(rotated_ham.ns[:ll+1+mm])
                        self.Rev_decay[n:n+rotated_ham.ns[ll],
                                       m:m+other_block.m] += \
                        np.sum(abs2(other_block.matrix), axis=0)

        return self.Rev_decay


    def construct_evolution_matrix(self, r, v, t=0.,
                                   default_axis=np.array([0., 0., 1.])):
        """
        Method to generate the rate equation evolution matrix.  Expects
        several arguments:
            r: a three-element vector specifying the position of interest
            v: a three-element vector specifying the velocity of interest
        """
        # What is the magnetic field?
        if self.tdepend['B']:
            B = self.magField.Field(r, t)
        else:
            B = self.magField.Field(r)

        # Calculate its magnitude:
        Bmag = np.linalg.norm(B, axis=0)

        # Calculate the Bhat direction:
        if Bmag > 1e-10:
            Bhat = B/Bmag
        else:
            Bhat = default_axis

        # Diagonalize at Bmag.  This is required if the Hamiltonian has any
        # non-diagonal elements of the Hamiltonian that are dependent on Bz.
        # This occurs, for example, with atoms in the large
        rotated_ham = self.hamiltonian.diag_static_field(Bmag)
        if not np.all(self.hamiltonian.diagonal):
            # Reconstruct the decay matrix to match this new field.
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
            kvecs = self.laserBeams[key].kvec(r, t)
            betas = self.laserBeams[key].beta(r, t)
            deltas = self.laserBeams[key].delta(t)

            projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

            # Loop through each laser beam driving this transition:
            for ll, (kvec, beta, proj, delta) in enumerate(zip(kvecs, betas, projs, deltas)):
                for ii in range(dijq.shape[1]):
                    for jj in range(dijq.shape[2]):
                        fijq = np.abs(np.dot(dijq[:, ii, jj], proj[::-1]))**2
                        if fijq > 0:
                            # Finally, calculate the scattering rate the polarization
                            # onto the appropriate basis:
                            """Rijl[ll, ii, jj] = beam.beta(r)/2*fijq/\
                            (1 + 4*(beam.delta - (H0[ng+jj, ng+jj] - H0[ii, ii]) -
                                    np.dot(kvec, v))**2)"""
                            self.Rijl[key][ll, ii, jj] = beta/2*\
                                fijq/(1 + 4*(-(E2[jj] - E1[ii]) + delta -
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


    def equilibrium_populations(self, r, v, t, **kwargs):
        """
        Returns the equilibrium values of the rate equation matrix Rev by
        singular matrix decomposition.
        """
        return_details = kwargs.pop('return_details', False)

        Rev, Rijl = self.construct_evolution_matrix(r, v, t, **kwargs)

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


    def force(self, r, t, Npop, return_details=True):
        F = np.zeros((3,))
        f = {}

        for key in self.laserBeams:
            f[key] = np.zeros((3, len(self.laserBeams[key].beam_vector)))

            ind = self.hamiltonian.laser_keys[key]
            n = sum(self.hamiltonian.ns[:ind[0]])
            m = sum(self.hamiltonian.ns[:ind[1]])
            for ll, beam in enumerate(self.laserBeams[key].beam_vector):
                # If kvec is callable, evaluate kvec:
                kvec = beam.kvec(r, t)

                for ii in range(self.Rijl[key].shape[1]):
                    for jj in range(self.Rijl[key].shape[2]):
                        if self.Rijl[key][ll, ii, jj]>0:
                            f[key][:, ll] += kvec*self.Rijl[key][ll, ii, jj]*\
                                (Npop[n+ii] - Npop[m+jj])

            F += np.sum(f[key], axis=1)

        fmag = np.array([0., 0., 0.])
        if self.include_mag_forces:
            gradBmag = self.magField.gradFieldMag(r)

            for ii, block in enumerate(np.diag(self.hamiltonian.blocks)):
                ind1 = int(np.sum(self.hamiltonian.ns[:ii]))
                ind2 = int(np.sum(self.hamiltonian.ns[:ii+1]))
                if self.hamiltonian.diagonal[ii]:
                    if isinstance(block, tuple):
                        fmag += np.sum(np.real(
                            block[1].matrix[1] @ Npop[ind1:ind2]
                            ))*gradBmag
                    elif isinstance(block, self.hamiltonian.vector_block):
                        fmag += np.sum(np.real(
                            block.matrix[1] @ Npop[ind1:ind2]
                            ))*gradBmag
                else:
                    if isinstance(block, tuple):
                        fmag += np.sum(np.real(
                            self.hamiltonian.U[ii].T @ block[1].matrix[1] @
                            self.hamiltonian.U[ii]) @ Npop[ind1:ind2])*gradBmag
                    elif isinstance(block, self.hamiltonian.vector_block):
                        fmag += np.sum(np.real(
                            self.hamiltonian.U[ii].T @ block.matrix[1] @
                            self.hamiltonian.U[ii]) @ Npop[ind1:ind2])*gradBmag

            F += fmag

        if return_details:
            return F, f, fmag
        else:
            return F

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
        free_axes = np.bitwise_not(kwargs.pop('freeze_axis', np.array([False, False, False])))
        random_recoil_flag = kwargs.pop('random_recoil', False)
        random_force_flag = kwargs.pop('random_recoil', False)
        recoil_velocity = kwargs.pop('recoil_velocity', 0.01)
        max_scatter_probability = kwargs.pop('max_scatter_probability', 0.1)
        progress_bar = kwargs.pop('progress_bar', False)
        record_force = kwargs.pop('record_force', False)

        if progress_bar:
            progress = progressBar()

        if record_force:
            ts = []
            Fs = []

        def motion(t, y):
            N = y[:-6]
            v = y[-6:-3]
            r = y[-3:]

            Rev, Rijl = self.construct_evolution_matrix(r, v, t)
            if not random_force_flag:
                if record_force:
                    F = self.force(r, t, N, return_details=True)

                    ts.append(t)
                    Fs.append(F)

                    F = F[0]
                else:
                    F = self.force(r, t, N, return_details=False)

                dydt = np.concatenate((Rev @ N,
                                       recoil_velocity*F*free_axes+
                                       self.constant_accel,
                                       v))
            else:
                dydt = np.concatenate((Rev @ N,
                                       self.constant_accel,
                                       v))

            if np.any(np.isnan(dydt)):
                raise ValueError('Enountered a NaN!')

            if progress_bar:
                progress.update(t/t_span[-1])

            return dydt

        def random_force(t, y, dt):
            scatter_dt_max = 1e11

            # Grab self.Rijl and sum over all ij to get
            for key in self.laserBeams:
                # Extract the pumping rate from each laser:
                Rl = np.sum(np.sum(self.Rijl[key],axis=2), axis=1)

                # Calculate the probability to scatter a photon from the laser:
                P = Rl*dt

                # Roll the dice N times, where $N=\sum(lasers)
                dice = np.random.rand(len(P))

                # Give them kicks!
                for ii, (P_i, dice_i) in enumerate(zip(P, dice)):
                    if dice_i<P_i:
                        y[-6:-3] += \
                        recoil_velocity*self.laserBeams[key].beam_vector[ii].kvec

                scatter_dt_max = min(new_dt_max, np.amax(max_scatter_probability/Rl))

            recoil_dt_max = random_recoil(t, y, dt)

            return min(recoil_dt_max, scatter_dt_max)


        def random_recoil(t, y, dt):
            # Calculate the probability that all excited states can decay.
            # $P_i = Gamma_i dt n_i$, where $n_i$ is the population in state $i$
            # TODO: currently gamma is considered to be unity:
            P = np.abs(y[self.hamiltonian.ns[0]:-6]*dt)

            # Roll the dice N times, where $N=\sum(n_i)
            dice = np.random.rand(len(P))

            # For any random number that is lower than P_i, add a recoil velocity.
            # TODO: There are potential problems in the way the kvector is defined.
            # The first is the k-vector is defined in the laser beam (not in the
            # Hamiltonian).  The second is that we should break this out according
            # to laser beam in case they do have different k-vectors.
            num_of_scatters = np.sum(dice<P)
            for ii in range(num_of_scatters):
                y[-6:-3] += recoil_velocity*(random_vector(free_axes)+random_vector(free_axes))

            new_dt_max = (max_scatter_probability/np.sum(P))*dt
            return (num_of_scatters, new_dt_max)

        y0 = np.concatenate((self.N0, self.v0, self.r0))
        if random_force_flag:
            self.sol = solve_ivp_random(motion, random_force, t_span, y0,
                                        max_step=0.01, **kwargs)
        elif random_recoil_flag:
            self.sol = solve_ivp_random(motion, random_recoil, t_span, y0,
                                        max_step=0.01, **kwargs)
        else:
            self.sol = solve_ivp(motion, t_span, y0, **kwargs)

        # Rearrange the solution:
        self.sol.N = self.sol.y[-6:]
        self.sol.v = self.sol.y[-6:-3]
        self.sol.r = self.sol.y[-3:]

        if record_force:
            f = interp1d(ts[:-1], np.array([f[0] for f in Fs[:-1]]).T)
            self.sol.F = f(self.sol.t)

            f = interp1d(ts[:-1], np.array([f[2] for f in Fs[:-1]]).T)
            self.sol.fmag = f(self.sol.t)

            self.sol.f = {}
            for key in Fs[0][1]:
                f = interp1d(ts[:-1], np.array([f[1][key] for f in Fs[:-1]]).T)
                self.sol.f[key] = f(self.sol.t)
                self.sol.f[key] = np.swapaxes(self.sol.f[key], 0, 1)

        del self.sol.y


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
                self.r0, self.v0, t=0, return_details=True, **kwargs
                )

            F_eq, f_eq, f_mag = self.force(self.r0, 0., N_eq)

        if return_details:
            return F_eq, f_eq, N_eq, Rijl, f_mag
        else:
            return F_eq

    def generate_force_profile(self, R, V, **kwargs):
        """
        Method that maps out the equilbirium pupming rate, populations,
        and forces:
        """
        name = kwargs.pop('name', None)
        progress_bar = kwargs.pop('progress_bar', None)

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams,
                                           self.hamiltonian)

        it = np.nditer([R[0], R[1], R[2], V[0], V[1], V[2]],
                       flags=['refs_ok', 'multi_index'],
                       op_flags=[['readonly'], ['readonly'], ['readonly'],
                                 ['readonly'], ['readonly'], ['readonly']])

        if progress_bar:
            progress = progressBar()

        for (x, y, z, vx, vy, vz) in it:
            # Construct the rate equations:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            # Update position (use multi_index), populations, and forces.
            self.set_initial_position_and_velocity(r, v)
            try:
                F, f, Neq, Rijl, f_mag = self.find_equilibrium_force(
                    return_details=True, **kwargs
                    )
            except:
                raise ValueError(
                    "Unable to find solution at " +\
                    "r=({0:0.2f},{1:0.2f},{2:0.2f})".format(x, y, z) + " and " +
                    "v=({0:0.2f},{1:0.2f},{2:0.2f})".format(vx, vy, vz)
                    )

            if progress_bar:
                progress.update((it.iterindex+1)/it.itersize)

            self.profile[name].store_data(it.multi_index, Neq, F, f, f_mag, Rijl)


class trap(rateeq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_eq = None


    def find_equilibrium_position(self, axes=[2], upper_lim=5., lower_lim=-5.,
                                  Npts=51, initial_search=True):
        if self.r_eq is None:
            self.r_eq = np.zeros((3,))

        # Next, find the equilibrium point in z, and evaluate derivatives there:
        r_eqi = np.zeros((3,))
        z = np.linspace(lower_lim, upper_lim, Npts)

        if initial_search:
            for axis in axes:
                v = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r = np.array([np.zeros(z.shape), np.zeros(z.shape), np.zeros(z.shape)])
                r[axis] = z

                default_axis=np.zeros((3,))
                default_axis[axis] = 1.
                self.generate_force_profile(r, v, name='root_search',
                                            default_axis=default_axis)

                z_possible = z[np.where(np.diff(np.sign(
                    self.profile['root_search'].F[axis]))<0)[0]]

                if z_possible.size>0:
                    if z_possible.size>1:
                        ind = np.argmin(z_possible**2)
                    else:
                        ind = 0
                    r_eqi[axis] = z_possible[ind]
                else:
                    r_eqi[axis] = np.nan

                del self.profile['root_search']

        #print('Initial guess: %s' % r_eqi[axes])
        if len(axes)>1:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return np.sum(F**2)

            if np.sum(np.isnan(r_eqi)) == 0:
                # Find the center of the trap:
                result = minimize(simple_wrapper, r_eqi[axes], method='SLSQP')
                if result.success:
                    self.r_eq[axes] = result.x
                else:
                    self.r_eq[axes] = np.nan
            else:
                self.r_eq = np.nan
        else:
            def simple_wrapper(r_changing):
                r_wrap = np.zeros((3,))
                r_wrap[axes] = r_changing

                self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
                F = self.find_equilibrium_force()

                return F[axes]

            if np.sum(np.isnan(r_eqi)) == 0:
                self.r_eq[axes] = fsolve(simple_wrapper, r_eqi[axes])[0]
            else:
                self.r_eq[axes] = np.nan

        return self.r_eq

    def trapping_frequencies(self, axes=[0, 2], r=None, eps=0.01):
        self.omega = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None:
            r = self.r_eq

        for axis in axes:
            if not np.isnan(r[axis]):
                rpmdri = np.tile(r, (2,1)).T
                rpmdri[axis, 1] += eps[axis]
                rpmdri[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(rpmdri[:, jj],
                                                           np.zeros((3,)))
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
            r = self.r_eq

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
