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

        # Set up two dictionaries that are useful for both random forces and
        # random recoils:
        self.decay_rates = {}
        self.decay_N_indices = {}

        # If the matrix is diagonal, we get to do something cheeky.  Let's just
        # construct the decay part of the evolution once:
        if np.all(self.hamiltonian.diagonal):
            self._calc_decay_comp_of_Rev(self.hamiltonian)

        # Save the recoil velocity for the relevant transitions:
        self.recoil_velocity = {}
        for key in self.hamiltonian.laser_keys:
            self.recoil_velocity[key] = \
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k']\
                /self.hamiltonian.mass

        # Reset the current solution to
        self.set_initial_position_and_velocity(r, v)

        # A dictionary to store the pumping rates.
        self.Rijl = {}
        
        # Set up a dictionary to store the profiles.
        self.profile = {}


    def _calc_decay_comp_of_Rev(self, rotated_ham):
        """
        Constructs the decay portion of the evolution matrix.

        Parameters
        ----------
        rotated_ham: pylcp.hamiltonian object
            The diagonalized hamiltonian with rotated d_q matrices
        """
        self.Rev_decay = np.zeros((self.hamiltonian.n, self.hamiltonian.n))

        # Go through each of the blocks and calculate the decay rate
        # contributions to the rate equations.  This would be simpler with
        # rotated_ham.make_full_matrices(), but I would imagine it would
        # be significantly slower.

        # Let's use the laser_keys dictionary to go through the non-zero d_q:
        for key in self.hamiltonian.laser_keys:
            # The index of the current d_q matrix:
            ind = rotated_ham.laser_keys[key]

            # Grab the block of interest:
            d_q_block = rotated_ham.blocks[ind]

            # The offset index for the lower states:
            noff = int(np.sum(rotated_ham.ns[:ind[0]]))
            # The offset index for the higher states:
            moff = int(np.sum(rotated_ham.ns[:ind[1]]))

            # The number of lower states:
            n = rotated_ham.ns[ind[0]]
            # The number of higher states:
            m = rotated_ham.ns[ind[1]]

            # Calculate the decay of the states attached to this block:
            gamma = d_q_block.parameters['gamma']
            k = d_q_block.parameters['k']

            # Let's see if we can avoid a for loop here:
#             for jj in range(rotated_ham.ns[ll]):
#                 self.Rev_decay[n+jj, n+jj] -= gamma*\
#                             np.sum(np.sum(abs2(
#                                 other_block.matrix[:, :, jj]
#                             )))
            # Save the decay rates out of the excited state:
            self.decay_rates[key] = gamma*np.sum(abs2(
                d_q_block.matrix[:, :, :]
            ), axis=(0,1))

            # Save the indices for the excited states of this d_q block
            # for the random_recoil function:
            self.decay_N_indices[key] = np.arange(moff, moff+m)

            # Add (more accurately, subtract) these decays to the evolution matrix:
            self.Rev_decay[(np.arange(moff, moff+m), np.arange(moff, moff+m))] -= \
            self.decay_rates[key]

            # Now calculate the decays into the lower state connected by this
            # d_q:
            self.Rev_decay[noff:noff+n, moff:moff+m] += \
                        gamma*np.sum(abs2(d_q_block.matrix), axis=0)

        return self.Rev_decay

        
    def _calc_pumping_rates(self, r, v, t, Bhat):
        """
        This method calculates the pumping rates for each laser beam:
        """
        for key in self.laserBeams:
            # Extract the relevant d_q matrix:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]
            d_q = self.hamiltonian.rotated_hamiltonian.blocks[ind].matrix
            gamma = self.hamiltonian.blocks[ind].parameters['gamma']

            # Extract the energies:
            E1 = np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[0],ind[0]].matrix)
            E2 = np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[1],ind[1]].matrix)

            E2, E1 = np.meshgrid(E2, E1)

            # Initialize the pumping matrix:
            self.Rijl[key] = np.zeros((len(self.laserBeams[key].beam_vector),) +
                                      d_q.shape[1:])

            # Grab the laser parameters:
            kvecs = self.laserBeams[key].kvec(r, t)
            betas = self.laserBeams[key].beta(r, t)
            deltas = self.laserBeams[key].delta(t)

            projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

            # Loop through each laser beam driving this transition:
            for ll, (kvec, beta, proj, delta) in enumerate(zip(kvecs, betas, projs, deltas)):
                fijq = np.abs(d_q[0]*proj[2] + d_q[1]*proj[1] +d_q[2]*proj[0])**2

                # Finally, calculate the scattering rate the polarization
                # onto the appropriate basis:
                self.Rijl[key][ll] = gamma*beta/2*\
                    fijq/(1 + 4*(-(E2 - E1) + delta - np.dot(kvec, v))**2/gamma**2)

    
    def _add_pumping_rates_to_Rev(self):
        # Now add the pumping rates into the rate equation propogation matrix:
        for key in self.laserBeams:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]

            n_off = sum(self.hamiltonian.rotated_hamiltonian.ns[:ind[0]])
            n = self.hamiltonian.rotated_hamiltonian.ns[ind[0]]
            m_off = sum(self.hamiltonian.rotated_hamiltonian.ns[:ind[1]])
            m = self.hamiltonian.rotated_hamiltonian.ns[ind[1]]

            # Sum over all lasers:
            Rij = np.sum(self.Rijl[key], axis=0)

            # Add the off diagonal components:
            self.Rev[n_off:n_off+n, m_off:(m_off+m)] += Rij
            self.Rev[m_off:(m_off+m), n_off:n_off+n] += Rij.T

            # Add the diagonal components.
            self.Rev[(np.arange(n_off, n_off+n), np.arange(n_off, n_off+n))] -= np.sum(Rij, axis=1)
            self.Rev[(np.arange(m_off, m_off+m), np.arange(m_off, m_off+m))] -= np.sum(Rij, axis=0)


    def construct_evolution_matrix(self, r, v, t=0., default_axis=np.array([0., 0., 1.])):
        """
        Constructs the
        
        Parameters
        ----------
            r: a three-element vector specifying the position of interest
            v: a three-element vector specifying the velocity of interest
        """
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
            
        # Diagonalize the hamiltonian at this location:
        self.hamiltonian.diag_static_field(Bmag)
            
        # Re-initialize the evolution matrix:
        self.Rev = np.zeros((self.hamiltonian.n, self.hamiltonian.n))
        
        if not np.all(self.hamiltonian.diagonal):
            # Reconstruct the decay matrix to match this new field.
            self.Rev_decay = self._calc_decay_comp_of_Rev(
                self.hamiltonian.rotated_hamiltonian
            )
            
        self.Rev += self.Rev_decay
        
        # Recalculate the pumping rates:
        Bhat = self._calc_pumping_rates(r, v, t, Bhat)
        
        # Add the pumping rates to the evolution matrix:
        self._add_pumping_rates_to_Rev()
                                         
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
            n_off = sum(self.hamiltonian.ns[:ind[0]])
            n = self.hamiltonian.ns[ind[0]]
            m_off = sum(self.hamiltonian.ns[:ind[1]])
            m = self.hamiltonian.ns[ind[1]]

            Ne, Ng = np.meshgrid(Npop[m_off:(m_off+m)], Npop[n_off:(n_off+n)], )
            
            for ll, beam in enumerate(self.laserBeams[key].beam_vector):
                kvec = beam.kvec(r, t)
                f[key][:, ll] += kvec*np.sum(self.Rijl[key][ll]*(Ng - Ne), axis=(0,1))

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
        random_force_flag = kwargs.pop('random_force', False)
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
                                       F*free_axes/self.hamiltonian.mass+
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
            total_P = 0
            num_of_scatters = 0

            # Go over all available keys:
            for key in self.laserBeams:
                # Extract the pumping rate from each laser:
                Rl = np.sum(self.Rijl[key], axis=(1,2))

                # Calculate the probability to scatter a photon from the laser:
                P = Rl*dt

                # Roll the dice N times, where $N=\sum(lasers)
                dice = np.random.rand(len(P))

                # Give them kicks!
                for ii in np.arange(len(Rl))[dice<P]:
                    num_of_scatters += 1
                    y[-6:-3] += self.laserBeams[key].beam_vector[ii].kvec(y[-3:], t)/\
                                self.hamiltonian.mass
                    # Can branch to a differe, lower state, but let's ignore that
                    # for the moment.
                    y[-6:-3] += self.recoil_velocity[key]*(random_vector(free_axes))

                total_P += np.sum(P)

            # Calculate a new maximum dt to make sure we evolve while not
            # exceeding dt max:
            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)


        def random_recoil(t, y, dt):
            num_of_scatters = 0
            total_P = 0.

            # Go over each block in the Hamiltonian and compute the decay:
            for key in self.decay_rates:
                P = dt*self.decay_rates[key]*y[self.decay_N_indices[key]]

                # Roll the dice N times, where $N=\sum(n_i)
                dice = np.random.rand(len(P))

                # For any random number that is lower than P_i, add a
                # recoil velocity.
                for ii in range(np.sum(dice<P)):
                    num_of_scatters += 1
                    y[-6:-3] += self.recoil_velocity[key]*(random_vector(free_axes)+
                                                           random_vector(free_axes))

                # Save the total probability of a scatter:
                total_P += np.sum(P)

            # Calculate a new maximum dt to make sure we evolve while not
            # exceeding dt max:
            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        y0 = np.concatenate((self.N0, self.v0, self.r0))
        if random_force_flag:
            self.sol = solve_ivp_random(motion, random_force, t_span, y0,
                                        initial_max_step=max_scatter_probability,
                                        **kwargs)
        elif random_recoil_flag:
            self.sol = solve_ivp_random(motion, random_recoil, t_span, y0,
                                        initial_max_step=max_scatter_probability,
                                        **kwargs)
        else:
            self.sol = solve_ivp(motion, t_span, y0, **kwargs)

        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        # Rearrange the solution:
        self.sol.N = self.sol.y[:-6]
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

        if r is None and self.r_eq is None:
            r = np.array([0., 0., 0.])
        elif r is None:
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

        if r is None and self.r_eq is None:
            r = np.array([0., 0., 0.])
        elif r is None:
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
