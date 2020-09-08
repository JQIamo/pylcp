import numpy as np
import copy
import time
import numba
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .integration_tools import solve_ivp_random
from .common import (progressBar, random_vector, spherical_dot,
                     cart2spherical, spherical2cart)
from .common import base_force_profile as force_profile
from .governingeq import governingeq

class heuristiceq(governingeq):
    """
    Heuristic force equation

    Parameters
    ----------
    laserBeams : dictionary of pylcp.laserBeams, pylcp.laserBeams, or list of pylcp.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.  It can
        be any of the following:
        - A dictionary of laserBeams: if this is the case, the keys of the
          laser beams should address the
    magField : pylcp.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleraiton to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    """
    def __init__(self, laserBeams, magField, a=np.array([0., 0., 0.]),
                 mass=100, gamma=1, k=1, r0=np.array([0., 0., 0.]),
                 v0=np.array([0., 0., 0.])):
        super().__init__(laserBeams, magField, a=a, r0=r0, v0=v0)

        # Finally, handle optional arguments:
        self.mass = mass
        self.gamma = gamma
        self.k = k

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


    def find_equilibrium_force(self, return_details=False):
        """
        Finds the equilibrium force at the initial position

        Parameters
        ----------
        return_details : boolean, optional
            If True, returns the forces from each laser and the scattering rate
            matrix.

        Returns
        -------
        F : array_like
            total equilibrium force experienced by the atom
        F_laser : array_like
            If return_details is True, the forces due to each laser.
        R : array_like
            The scattering rate matrix.
        """
        F, F_laser = self.force(self.r0, self.v0, t=0.)

        if return_details:
            return F, F_laser, self.R
        else:
            return F


    def generate_force_profile(self, R, V, name=None, progress_bar=False):
        """
        Map out the equilibrium force vs. position and velocity

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector.  First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math`z` components,
            repsectively.
        V : array_like, shape(3, ...)
            Velocity vector.  First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math`v_z` components,
            repsectively.
        name : str, optional
            Name for the profile.  Stored in profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.
        progress_bar : boolean, optional
            Displays a progress bar as the proceeds.  Default: False

        Returns
        -------
        profile : pylcp.common.base_force_profile
            Resulting force profile.
        """
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

        return self.profile[name]
