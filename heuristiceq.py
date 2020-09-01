import numpy as np
import copy
import time
import numba
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .fields import laserBeams, magField
from .integration_tools import solve_ivp_random
from .common import (progressBar, random_vector, spherical_dot,
                     cart2spherical, spherical2cart, governingeq)
from .common import base_force_profile as force_profile

class heuristiceq(governingeq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Do argument checking specific
        if len(args) < 2:
            raise ValueError('You must specify laserBeams and magField')
        elif len(args) == 2:
            self.constant_accel = np.array([0., 0., 0.])
        elif len(args) == 3:
            if not isinstance(args[2], np.ndarray):
                raise TypeError('Constant acceleration must be an numpy array.')
            elif args[2].size != 3:
                raise ValueError('Constant acceleration must have length 3.')
            else:
                self.constant_accel = args[2]
        else:
            raise ValueError('No more than four positional arguments accepted.')

        # Add lasers:
        self.laserBeams = {} # Laser beams are meant to be dictionary,
        if isinstance(args[0], list):
            self.laserBeams['g->e'] = copy.copy(laserBeams(args[0])) # Assume label is g->e
        elif isinstance(args[0], laserBeams):
            self.laserBeams['g->e'] = copy.copy(args[0]) # Again, assume label is g->e
        elif isinstance(args[0], dict):
            for key in args[0].keys():
                if not key is 'g->e':
                    raise KeyError('laserBeam dictionary should only contain ' +
                                   'a single key of \'g->e\' for the heutisticeq.')
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

        # Finally, handle optional arguments:
        self.mass = kwargs.pop('mass', 100)
        self.gamma = kwargs.pop('gamma', 1)
        self.k = kwargs.pop('k', 1)

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol = None

        # Make some variables to store F, F_laser, and R_sc:
        self.F = np.array([0., 0., 0.])
        self.F_laser = {}
        self.F_laser['g->e'] = np.zeros((3, self.laserBeams['g->e'].num_of_beams))
        self.R = np.zeros((self.laserBeams['g->e'].num_of_beams, ))

    def scattering_rate(self, r, v, t, return_kvecs=False):
        B = self.magField.Field(r, t)
        Bmag = np.linalg.norm(B)
        if Bmag==0:
            Bhat = np.array([0., 0., 1.])
        else:
            Bhat = B/np.linalg.norm(B)

        kvecs = self.laserBeams['g->e'].kvec(r, t)
        betas = self.laserBeams['g->e'].beta(r, t)
        pols = self.laserBeams['g->e'].project_pol(Bhat, r, t)
        deltas = self.laserBeams['g->e'].delta(t)

        totbeta = np.sum(betas)

        for ii, (kvec, beta, pol, delta) in enumerate(zip(kvecs, betas, pols, deltas)):
            self.R[ii] = 0.
            polsqrd = np.abs(pol)**2
            for (q, pol_i) in zip(np.array([-1., 0., 1.]), polsqrd):
                self.R[ii] += self.gamma/2*beta*pol_i/\
                (1+ totbeta + 4*(delta - np.dot(kvec, v) - q*Bmag)**2/self.gamma**2)

        if return_kvecs:
            return self.R, kvecs
        else:
            return self.R

    def force(self, r, v, t):
        R, kvecs = self.scattering_rate(r, v, t, return_kvecs=True)

        self.F_laser['g->e'] = (kvecs*R[:, np.newaxis]).T
        self.F = np.sum(self.F_laser['g->e'], axis=1)

        return self.F, self.F_laser

    def evolve_motion(self, t_span, **kwargs):
        free_axes = np.bitwise_not(kwargs.pop('freeze_axis', [False, False, False]))
        random_recoil_flag = kwargs.pop('random_recoil', False)
        random_force_flag = kwargs.pop('random_force', False)
        max_scatter_probability = kwargs.pop('max_scatter_probability', 0.1)
        progress_bar = kwargs.pop('progress_bar', False)

        if progress_bar:
            progress = progressBar()

        def dydt(t, y):
            if progress_bar:
                progress.update(t/t_span[1])

            F, Flaser = self.force(y[3:6], y[0:3], t)

            return np.concatenate((
                1/self.mass*F*free_axes + self.constant_accel,
                y[0:3]
                ))

        def dydt_random_force(t, y):
            if progress_bar:
                progress.update(t/t_span[1])

            return np.concatenate((self.constant_accel, y[0:3]))

        def random_recoil(t, y, dt):
            num_of_scatters = 0
            total_P = np.sum(self.R)*dt
            if np.random.rand(1)<total_P:
                y[0:3] += self.k/self.mass*(random_vector(free_axes)+
                                            random_vector(free_axes))
                num_of_scatters += 1

            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        def random_force(t, y, dt):
            R, kvecs = self.scattering_rate(y[3:6], y[0:3], t, return_kvecs=True)

            num_of_scatters = 0
            for kvec in kvecs[np.random.rand(len(R))<R*dt]:
                y[-6:-3] += kvec/self.mass
                y[-6:-3] += self.k/self.mass*random_vector(free_axes)

                num_of_scatters += 1

            total_P = np.sum(self.R)*dt
            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        if not random_recoil_flag and not random_force_flag:
            self.sol = solve_ivp(
                dydt, t_span, np.concatenate((self.v0, self.r0)),
                **kwargs)
        elif random_force_flag:
            self.sol = solve_ivp_random(
                dydt_random_force, random_force, t_span,
                np.concatenate((self.v0, self.r0)),
                initial_max_step=max_scatter_probability,
                **kwargs)
        else:
            self.sol = solve_ivp_random(
                dydt, random_recoil, t_span,
                np.concatenate((self.v0, self.r0)),
                initial_max_step=max_scatter_probability,
                **kwargs
                )

        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        self.sol.r = self.sol.y[3:]
        self.sol.v = self.sol.y[:3]


    def find_equilibrium_force(self, **kwargs):
        return_details = kwargs.pop('return_details', False)

        F, F_laser = self.force(self.r0, self.v0, t=0.)

        if return_details:
            return F, F_laser, self.R
        else:
            return F

    def generate_force_profile(self, R, V,  **kwargs):
        """
        Method that maps out the equilbirium forces:
        """
        name = kwargs.pop('name', None)
        progress_bar = kwargs.pop('progress_bar', False)
        deltat_r = kwargs.pop('deltat_r', None)
        deltat_v = kwargs.pop('deltat_v', None)
        deltat_tmax = kwargs.pop('deltat_tmax', np.inf)
        initial_rho = kwargs.pop('initial_rho', 'rateeq')

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, None)

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

            F, F_laser = self.force(r, v, 0.)

            self.profile[name].store_data(it.multi_index, None, F, F_laser,
                                          np.zeros((3,)))

            if progress_bar:
                progress.update((it.iterindex+1)/it.itersize)
