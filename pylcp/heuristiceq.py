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

    The heuristic equation governs the atom or molecule as if it has a single
    transition between an :math:`F=0` ground state to an :math:`F'=1` excited
    state.

    Parameters
    ----------
    laserBeams : dictionary of pylcp.laserBeams, pylcp.laserBeams, or list of pylcp.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.  It can
        be any of the following:

            * A dictionary of pylcp.laserBeams: if this is the case, the keys of
              the dictionary should match available :math:`d^{nm}` matrices
              in the pylcp.hamiltonian object.  The key structure should be
              `n->m`.  Here, it must be `g->e`.
            * pylcp.laserBeams: a single set of laser beams is assumed to
              address the transition `g->e`.
            * a list of pylcp.laserBeam: automatically promoted to a
              pylcp.laserBeams object assumed to address the transtion `g->e`.

    magField : pylcp.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleraiton to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    r0 : array_like, shape (3,), optional
        Initial position of the atom or molecule.  Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity of the atom or molecule.  Default: [0., 0., 0.]
    mass : float, optional
        Mass of the atom or molecule. Default: 100
    gamma : float, optional
        Decay rate of the single transition in the atom or molecule. Default: 1
    k : float, optional
        Magnitude of the k vector for the single transition in the atom or
        molecule. Default: 1
    """
    def __init__(self, laserBeams, magField, a=np.array([0., 0., 0.]),
                 mass=100, gamma=1, k=1, r0=np.array([0., 0., 0.]),
                 v0=np.array([0., 0., 0.])):
        super().__init__(laserBeams, magField, a=a, r0=r0, v0=v0)

        # Check to make sure the laserBeams dictionary has only one key:
        for key in self.laserBeams:
            if key != 'g->e':
                print(key)
                raise KeyError('laserBeam dictionary should only contain ' +
                                'a single key of \'g->e\' for the heutisticeq.')

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
        """
        Calculates the scattering rate

        Parameters
        ----------
        r : array_like
            Position at which to calculate the force
        v : array_like
            Velocity at which to calculate the force
        t : float
            Time at which to calculate the force
        return_kvecs : bool
            If true, returns both the scattering rate and the k-vecotrs from
            the lasers.

        Returns
        -------
        R : array_like
            Array of scattering rates associated with the lasers driving the
            transition.
        kvecs : array_like
            If return_kvecs is True, the k-vectors of each of the lasers.  This
            is used in heuristiceq.force, where it calls this function to
            calculate the scattering rate first.  By returning the k-vectors
            with the scattering rates, it prevents the need of having to
            recompute the k-vectors again.
        """
        B = self.magField.Field(r, t)
        Bmag = np.linalg.norm(B)
        if Bmag==0:
            Bhat = np.array([0., 0., 1.])
        else:
            Bhat = B/np.linalg.norm(B)

        kvecs = self.laserBeams['g->e'].kvec(r, t)
        intensities = self.laserBeams['g->e'].intensity(r, t)
        pols = self.laserBeams['g->e'].project_pol(Bhat, r, t)
        deltas = self.laserBeams['g->e'].delta(t)

        totintensity = np.sum(intensities)

        for ii, (kvec, intensity, pol, delta) in enumerate(zip(kvecs, intensities, pols, deltas)):
            self.R[ii] = 0.
            polsqrd = np.abs(pol)**2
            for (q, pol_i) in zip(np.array([-1., 0., 1.]), polsqrd):
                self.R[ii] += self.gamma/2*intensity*pol_i/\
                (1+ totintensity + 4*(delta - np.dot(kvec, v) - q*Bmag)**2/self.gamma**2)

        if return_kvecs:
            return self.R, kvecs
        else:
            return self.R

    def force(self, r, v, t):
        """
        Calculates the instantaneous force

        Parameters
        ----------
        r : array_like
            Position at which to calculate the force
        v : array_like
            Velocity at which to calculate the force
        t : float
            Time at which to calculate the force

        Returns
        -------
        F : array_like
            total equilibrium force experienced by the atom
        F_laser : dictionary of array_like
            If return_details is True, the forces due to each laser, indexed
            by the manifold the laser addresses.  The dictionary is keyed by
            the transition driven, and individual lasers are in the same order
            as in the pylcp.laserBeams object used to create the governing
            equation.
        """
        R, kvecs = self.scattering_rate(r, v, t, return_kvecs=True)

        self.F_laser['g->e'] = (kvecs*R[:, np.newaxis]).T
        self.F = np.sum(self.F_laser['g->e'], axis=1)

        return self.F, self.F_laser

    def evolve_motion(self, t_span, freeze_axis=[False, False, False],
                      random_recoil=False, random_force=False,
                      max_scatter_probability=0.1, progress_bar=False,
                      rng=np.random.default_rng(), **kwargs):
        """
        Evolve the motion of the atom in time.

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        freeze_axis : list of boolean
            Freeze atomic motion along the specified axis.
            Default: [False, False, False]
        random_recoil : boolean
            Allow the atom to randomly recoil from scattering events.
            Default: False
        random_force : boolean
            Rather than calculating the force using the heuristieq.force() method,
            use the calculated scattering rates from each of the laser beam
            to randomly add photon absorption events that cause the atom to
            recoil randomly from the laser beam(s).
            Default: False
        max_scatter_probability : float
            When undergoing random recoils, this sets the maximum time step such
            that the maximum scattering probability is less than or equal to
            this number during the next time step.  Default: 0.1
        progress_bar : boolean
            If true, show a progress bar as the calculation proceeds.
            Default: False
        rng : numpy.random.Generator()
            A properly-seeded random number generator.  Default: calls
            ``numpy.random.default.rng()``
        **kwargs :
            Additional keyword arguments get passed to solve_ivp_random, which
            is what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * v: atomic velocity
                * r: atomic position

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
        free_axes = np.bitwise_not(freeze_axis)

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

        def random_recoil_func(t, y, dt):
            num_of_scatters = 0
            total_P = np.sum(self.R)*dt
            if rng.random(1)<total_P:
                y[0:3] += self.k/self.mass*(random_vector(rng, free_axes)+
                                            random_vector(rng, free_axes))
                num_of_scatters += 1

            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        def random_force_func(t, y, dt):
            R, kvecs = self.scattering_rate(y[3:6], y[0:3], t, return_kvecs=True)

            num_of_scatters = 0
            for kvec in kvecs[np.random.rand(len(R))<R*dt]:
                y[-6:-3] += kvec/self.mass
                y[-6:-3] += self.k/self.mass*random_vector(free_axes)

                num_of_scatters += 1

            total_P = np.sum(self.R)*dt
            new_dt_max = (max_scatter_probability/total_P)*dt

            return (num_of_scatters, new_dt_max)

        if random_force:
            self.sol = solve_ivp_random(
                dydt_random_force, random_force_func, t_span,
                np.concatenate((self.v0, self.r0)),
                initial_max_step=max_scatter_probability,
                **kwargs
                )
        elif random_recoil:
            self.sol = solve_ivp_random(
                dydt, random_recoil_func, t_span,
                np.concatenate((self.v0, self.r0)),
                initial_max_step=max_scatter_probability,
                **kwargs
                )
        else:
            self.sol = solve_ivp(
                dydt, t_span, np.concatenate((self.v0, self.r0)),
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
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            repsectively.
        V : array_like, shape(3, ...)
            Velocity vector.  First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
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
