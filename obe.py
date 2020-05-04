"""
Tools for solving the OBE for laser cooling
author: spe
"""
import numpy as np
import copy
import time
import numba
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from .rateeq import rateeq
from .fields import laserBeams, magField
from .integration_tools import solve_ivp_random
from .common import (printProgressBar, random_vector, spherical_dot,
                     cart2spherical, spherical2cart, base_force_profile)

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@numba.jit(nopython=True)
def dot(A, x):
    return A @ x

@numba.jit(nopython=True)
def dot_and_add(A, x, b):
    b += A @ x


class force_profile(base_force_profile):
    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.iterations = np.zeros(self.R[0].shape, dtype='int64')
        self.fq = {}
        for key in laserBeams:
            self.fq[key] = np.zeros(self.R.shape + (3, len(laserBeams[key].beam_vector)))

    def store_data(self, ind, Neq, F, F_laser, F_mag, iterations, F_laser_q):
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for jj in range(3):
            for key in F_laser_q:
                self.fq[key][(jj,) + ind] = F_laser_q[key][jj]

        self.iterations[ind] = iterations


class obe():
    """
    The class optical bloch equations prduces a set of optical Bloch equations
    for a given position and velocity and provides methods for
    solving them appropriately.
    """
    def __init__(self, *args, **kwargs):
        """
        construct_optical_bloch_eqns: this function takes in a hamiltonian, a
        set of laserBeams, and a magField function and an internal hamiltonian
        and sets up the optical bloch equations.  Arguments:
            r: position at which to evaluate the magnetic field.
            v: 3-vector velocity of the atom or molecule.
            laserBeams: a dictionary of laserBeams that say which fields drive
                which transitions in the block diagonal hamiltonian.
            magField: a function that defines the magnetic field.
            hamiltonian: the internal hamiltonian of the particle as defined by
                the hamiltonian class.
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

        # Next, check to see if there is consistency in k:
        self.__check_consistency_in_lasers_and_d_q()

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

        self.transform_into_re_im = kwargs.pop('transform_into_re_im', False)
        use_sparse_matrices = kwargs.pop('use_sparse_matrices', None)
        if use_sparse_matrices is None:
            if self.hamiltonian.n>10: # Generally offers a performance increase
                self.use_sparse_matrices = True
            else:
                self.use_sparse_matrices = False
        else:
            self.use_sparse_matrices = use_sparse_matrices
        self.include_mag_forces = kwargs.pop('include_mag_forces', True)

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol = None

        """
        There will be time-dependent and time-independent components of the optical
        Bloch equations.  The time-independent parts are related to spontaneous
        emission, applied magnetic field, and the zero-field Hamiltonian.  We
        compute the latter-two directly from the commuatator.
        """

        # Build the matricies that control evolution:
        self.ev_mat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        # If necessary, transform the evolution matrices:
        if self.transform_into_re_im:
            self.__transform_ev_matrices()

        if self.use_sparse_matrices:
            self.__convert_to_sparse()

        # Finally, update the position and velocity:
        self.set_initial_position_and_velocity(r, v)


    def __check_consistency_in_lasers_and_d_q(self):
        # Check that laser beam keys and Hamiltonian keys match.
        for laser_key in self.laserBeams.keys():
            if not laser_key in self.hamiltonian.laser_keys.keys():
                raise ValueError('laserBeams dictionary keys %s ' % laser_key +
                                 'does not have a corresponding key the '+
                                 'Hamiltonian d_q.')
                

    def __density_index(self, ii, jj):
        """
        This function returns the index in the rho vector that corresponds to element rho_{ij}.  If
        """
        return ii + jj*self.hamiltonian.n


    def __build_coherent_ev_submatrix(self, H):
        """
        This method builds the coherent evolution based on a submatrix of the
        Hamiltonian H.  In practice, one must be careful about commutators if
        one breaks up the Hamiltonian.
        """
        ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                               dtype='complex128')

        for ii in range(self.hamiltonian.n):
            for jj in range(self.hamiltonian.n):
                for kk in range(self.hamiltonian.n):
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(ii, kk)] += 1j*H[kk, jj]
                    ev_mat[self.__density_index(ii, jj),
                           self.__density_index(kk, jj)] -= 1j*H[ii, kk]

        return ev_mat


    def __build_coherent_ev(self):
        self.ev_mat['H0'] = self.__build_coherent_ev_submatrix(
            self.hamiltonian.H_0
        )

        self.ev_mat['B'] = [None]*3
        for q in range(3):
            self.ev_mat['B'][q] = self.__build_coherent_ev_submatrix(
                self.hamiltonian.mu_q[q]
            )
        self.ev_mat['B'] = np.array(self.ev_mat['B'])

        self.ev_mat['d_q'] = {}
        self.ev_mat['d_q*'] = {}
        for key in self.laserBeams.keys():
            self.ev_mat['d_q'][key] = [None]*3
            self.ev_mat['d_q*'][key] = [None]*3
            for q in range(3):
                self.ev_mat['d_q'][key][q] = self.__build_coherent_ev_submatrix(
                    self.hamiltonian.d_q_bare[key][q]
                )
                self.ev_mat['d_q*'][key][q] = self.__build_coherent_ev_submatrix(
                    self.hamiltonian.d_q_star[key][q]
                )
            self.ev_mat['d_q'][key] = np.array(self.ev_mat['d_q'][key])
            self.ev_mat['d_q*'][key] = np.array(self.ev_mat['d_q*'][key])


    def __build_decay_ev(self):
        self.ev_mat['decay'] = np.zeros((self.hamiltonian.n**2,
                                         self.hamiltonian.n**2),
                                         dtype='complex128')

        d_q = self.hamiltonian.d_q

        # Let's first do a check of the decay rates.  We want to make sure
        # that all states in a given manifold are, in fact, decaying at the
        # same rate.
        decay_rates = np.zeros((self.hamiltonian.blocks.shape[0],))
        for ll in range(1, self.hamiltonian.blocks.shape[0]):
            this_manifold = range(sum(self.hamiltonian.ns[:ll]),
                                  sum(self.hamiltonian.ns[:ll+1]))
            # We first check to make sure the decay rate is the same for all
            # states out of the manifold.
            rates = [np.sum(np.sum(abs2(d_q[:, :ii, ii]))) for ii in this_manifold]
            rate = np.mean(rates)
            if not np.allclose(rates, rate, atol=1e-7, rtol=1e-5):
                raise ValueError('Decay rates are not equal for all states in '+
                                 'manifold #%d' % ll)
            else:
                decay_rates[ll] = rate

        # Now we start building the evolution matrices.
        # Decay into a manifold.
        for ll in range(self.hamiltonian.blocks.shape[0]-1):
            this_manifold = range(sum(self.hamiltonian.ns[:ll]),
                                  sum(self.hamiltonian.ns[:ll+1]))
            all_higher_manifolds = range(sum(self.hamiltonian.ns[:ll+1]),
                                         sum(self.hamiltonian.ns))
            for ii in this_manifold:
                for jj in this_manifold:
                    for kk in all_higher_manifolds:
                        for ll in all_higher_manifolds:
                            for q in range(3):
                                self.ev_mat['decay'][self.__density_index(ii, jj),
                                                     self.__density_index(kk, ll)] +=\
                                d_q[q, ii, kk]*d_q[q, ll, jj]

        # Decay out of a manifold.  Each state and coherence in the manifold
        # decays with whatever the decay rate is.  In the present case, the
        # state $i$ decays with sum(d_q[:, :ii, ii])**2.
        for ll in range(1, self.hamiltonian.blocks.shape[0]):
            this_manifold = range(sum(self.hamiltonian.ns[:ll]),
                                  sum(self.hamiltonian.ns[:ll+1]))
            for ii in this_manifold:
                for jj in this_manifold:
                    self.ev_mat['decay'][self.__density_index(ii, jj),
                                         self.__density_index(ii, jj)] = -decay_rates[ll]

        # Coherences decay with the average decay rate out of the manifold
        # and into the manifold.
        for ll in range(self.hamiltonian.blocks.shape[0]-1):
            for mm in range(ll+1, self.hamiltonian.blocks.shape[0]):
                this_manifold = range(sum(self.hamiltonian.ns[:ll]),
                                      sum(self.hamiltonian.ns[:ll+1]))
                other_manifold = range(sum(self.hamiltonian.ns[:mm]),
                                       sum(self.hamiltonian.ns[:mm+1]))
                for ii in this_manifold:
                    for jj in other_manifold:
                        self.ev_mat['decay'][self.__density_index(ii, jj),
                                             self.__density_index(ii, jj)] = \
                        -(decay_rates[ll]+decay_rates[mm])/2
                        self.ev_mat['decay'][self.__density_index(jj, ii),
                                             self.__density_index(jj, ii)] = \
                        -(decay_rates[ll]+decay_rates[mm])/2


    def __build_transform_matrices(self):
        self.U = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                     dtype='complex128')
        self.Uinv = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                        dtype='complex128')

        for ii in range(self.hamiltonian.n):
            self.U[self.__density_index(ii, ii),
                   self.__density_index(ii, ii)] = 1.
            self.Uinv[self.__density_index(ii, ii),
                      self.__density_index(ii, ii)] = 1.

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                    self.U[self.__density_index(ii, jj),
                           self.__density_index(ii, jj)] = 1.
                    self.U[self.__density_index(ii, jj),
                           self.__density_index(jj, ii)] = 1j

                    self.U[self.__density_index(jj, ii),
                           self.__density_index(ii, jj)] = 1.
                    self.U[self.__density_index(jj, ii),
                           self.__density_index(jj, ii)] = -1j

        for ii in range(self.hamiltonian.n):
            for jj in range(ii+1, self.hamiltonian.n):
                    self.Uinv[self.__density_index(ii, jj),
                              self.__density_index(ii, jj)] = 0.5
                    self.Uinv[self.__density_index(ii, jj),
                              self.__density_index(jj, ii)] = 0.5

                    self.Uinv[self.__density_index(jj, ii),
                              self.__density_index(ii, jj)] = -0.5*1j
                    self.Uinv[self.__density_index(jj, ii),
                              self.__density_index(jj, ii)] = +0.5*1j


    def __transform_ev_matrix(self, ev_mat):
        if not hasattr(self, 'U'):
            self.__build_transform_matrices()

        ev_mat_new = self.Uinv @ ev_mat @ self.U

        # This should remove the imaginary component.
        if np.allclose(np.imag(ev_mat_new), 0):
            return np.real(ev_mat_new)
        else:
            raise ValueError('Something went dreadfully wrong.')


    def __transform_ev_matrices(self):
        self.ev_mat['decay'] = self.__transform_ev_matrix(self.ev_mat['decay'])
        self.ev_mat['H0'] = self.__transform_ev_matrix(self.ev_mat['H0'])

        self.ev_mat['reE'] = {}
        self.ev_mat['imE'] = {}
        for key in self.ev_mat['d_q'].keys():
            self.ev_mat['reE'][key] = np.array([self.__transform_ev_matrix(
                self.ev_mat['d_q'][key][jj] + self.ev_mat['d_q*'][key][jj]
                ) for jj in range(3)])
            # Unclear why the following works, I calculate that there should
            # be a minus sign out front.
            self.ev_mat['imE'][key] = np.array([self.__transform_ev_matrix(
                1j*(self.ev_mat['d_q'][key][jj] - self.ev_mat['d_q*'][key][jj])
                ) for jj in range(3)])

        # Transform Bq back into Bx, By, and Bz (making it real):
        self.ev_mat['B'] = spherical2cart(self.ev_mat['B'])

        for jj in range(3):
            self.ev_mat['B'][jj] = self.__transform_ev_matrix(self.ev_mat['B'][jj])
        self.ev_mat['B'] = np.real(self.ev_mat['B'])

        del self.ev_mat['d_q']
        del self.ev_mat['d_q*']


    def __convert_to_sparse(self):
        def convert_based_on_shape(matrix):
            # Vector:
            if matrix.shape == (3, self.hamiltonian.n**2, self.hamiltonian.n**2):
                new_list = [None]*3
                for jj in range(3):
                    new_list[jj] = sparse.csr_matrix(matrix[jj])

                return new_list
            # Scalar:
            else:
                return sparse.csr_matrix(matrix)

        for key in self.ev_mat:
            if isinstance(self.ev_mat[key], dict):
                for subkey in self.ev_mat[key]:
                    self.ev_mat[key][subkey] = convert_based_on_shape(
                        self.ev_mat[key][subkey]
                        )
            else:
                self.ev_mat[key] = convert_based_on_shape(self.ev_mat[key])


    def set_initial_position_and_velocity(self, r0, v0):
        self.set_initial_position(r0)
        self.set_initial_velocity(v0)

    def set_initial_position(self, r0):
        self.r0 = r0
        self.sol = None

    def set_initial_velocity(self, v0):
        self.v0 = v0
        self.sol = None

    def set_initial_rho(self, rho0):
        if np.any(np.isnan(rho0)) or np.any(np.isinf(rho0)):
            raise ValueError('rho0 has NaNs or Infs!')

        if self.transform_into_re_im and rho0.dtype is np.dtype('complex128'):
            self.rho0 = np.real(rho0)
        elif (not self.transform_into_re_im and
              not rho0.dtype is np.dtype('complex128')):
            self.rho0 = rho0.astype('complex128')
        else:
            self.rho0 = rho0

    def set_initial_rho_equally(self):
        if self.transform_into_re_im:
            self.rho0 = np.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = np.zeros((self.hamiltonian.n**2,), dtype='complex128')

        for jj in range(self.hamiltonian.ns[0]):
            self.rho0[self.__density_index(jj, jj)] = 1/self.hamiltonian.ns[0]

    def set_initial_rho_from_populations(self, Npop):
        if self.transform_into_re_im:
            self.rho0 = np.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = np.zeros((self.hamiltonian.n**2,), dtype='complex128')

        if len(Npop) != self.hamiltonian.n:
            raise ValueError('Npop has only %d entries for %d states.' %
                             (len(Npop), self.hamiltonian.n))
        if np.any(np.isnan(Npop)) or np.any(np.isinf(Npop)):
            raise ValueError('Npop has NaNs or Infs!')

        Npop = Npop/np.sum(Npop) # Just make sure it is normalized.
        for jj in range(self.hamiltonian.n):
            self.rho0[self.__density_index(jj, jj)] = Npop[jj]

    def set_initial_rho_from_rateeq(self):
        if not hasattr(self, 'rateeq'):
            self.rateeq = rateeq(self.laserBeams, self.magField, self.hamiltonian)
        Neq = self.rateeq.equilibrium_populations(self.r0, self.v0, t=0)
        self.set_initial_rho_from_populations(Neq)


    def full_OBE_ev_scratch(self, r, t):
        """
        This function calculates the OBE evolution matrix at position t and r
        from scratch, first computing the full Hamiltonian, then the
        OBE evolution matrix computed via commutators, then adding in the decay
        matrix evolution.

        If Bq is None, it will compute Bq
        """
        Eq = {}
        for key in self.laserBeams.keys():
            Eq[key] = self.laserBeams[key].total_electric_field(r, t)

        B = self.magField.Field(r, t)
        Bq = cart2spherical(B)

        H = self.hamiltonian.return_full_H(Bq, Eq)
        ev_mat = self.__build_coherent_ev_submatrix(H)

        if self.transform_into_re_im:
            return self.__transform_ev_matrix(ev_mat + self.ev_mat['decay'])
        else:
            return ev_mat + self.ev_mat['decay']


    def full_OBE_ev(self, r, t):
        """
        This function calculates the OBE evolution matrix by assembling
        pre-stored versions of the component matries.  This should be
        significantly faster than full_OBE_ev_scratch, but it may suffer bugs
        in the evolution that full_OBE_ev_scratch will not.

        If Bq is None, it will compute Bq based on r, t
        """
        ev_mat = self.ev_mat['decay'] + self.ev_mat['H0']

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii in range(3):
                    if np.abs(Eq[ii])>1e-10:
                        ev_mat -= 0.5*np.real(Eq[ii])*self.ev_mat['reE'][key][ii]
                        ev_mat -= 0.5*np.imag(Eq[ii])*self.ev_mat['imE'][key][ii]
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii in range(3):
                    if np.abs(Eq[ii])>1e-10:
                        ev_mat -= 0.5*np.conjugate(Eq[ii])*self.ev_mat['d_q'][key][ii]
                        ev_mat -= 0.5*Eq[ii]*self.ev_mat['d_q*'][key][ii]

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(Bq[ii])>1e-10:
                    drhodt -= self.ev_mat['B'][ii]*B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[2-ii])>1e-10:
                    drhodt -= (-1)**np.abs(q)*self.ev_mat['B'][ii]*Bq[2-ii] @ rho

        return ev_mat


    def drhodt(self, r, t, rho):
        """
        It is MUCH more efficient to do matrix vector products and add the
        results together rather than to add the matrices together (as above)
        and then do the dot.  It is also most efficient to avoid doing useless
        math if the applied field is zero.
        """
        drhodt = (self.ev_mat['decay'] @ rho) + (self.ev_mat['H0'] @ rho)

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii, q in enumerate(np.arange(-1., 2., 1)):
                    if np.abs(Eq[2-ii])>1e-10:
                        drhodt -= (0.5*(-1.)**q*np.real(Eq[2-ii])*
                                   (self.ev_mat['reE'][key][ii] @ rho))
                        drhodt -= (0.5*(-1.)**q*np.imag(Eq[2-ii])*
                                   (self.ev_mat['imE'][key][ii] @ rho))
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii, q in enumerate(np.arange(-1., 2., 1)):
                    if np.abs(Eq[2-ii])>1e-10:
                        drhodt -= (0.5*(-1.)**q*Eq[2-ii]*
                                   (self.ev_mat['d_q'][key][ii] @ rho))
                        drhodt -= (0.5*(-1.)**q*np.conjugate(Eq[2-ii])*
                                   (self.ev_mat['d_q*'][key][ii] @ rho))

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(B[ii])>1e-10:
                    drhodt -= self.ev_mat['B'][ii]*B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[2-ii])>1e-10:
                    drhodt -= (-1)**np.abs(q)*self.ev_mat['B'][ii]*Bq[2-ii] @ rho

        return drhodt


    def evolve_density(self, t_span, **kwargs):
        """
        This function evolves the optical bloch equations for some period of
        time.  Any initial velocity is kept constant while the atoms potentially
        moves through the light field.  This function is therefore useful in
        determining average forces.  It is analogous to evolve populations in
        the rateeq class.

        Any additional keyword arguments get passed to solve_ivp, which is
        what actually does the integration.
        """
        a = np.zeros((3,))

        def dydt(t, y):
            return np.concatenate((self.drhodt(y[-3:], t, y[:-6]), a, y[-6:-3]))

        self.sol = solve_ivp(dydt, t_span,
                             np.concatenate((self.rho0, self.v0, self.r0)),
                             **kwargs)


    def evolve_motion(self, t_span, **kwargs):
        """
        This function evolves the optical bloch equations for some period of
        time, with all their potential glory!
        """
        free_axes = np.bitwise_not(kwargs.pop('freeze_axis', [False, False, False]))
        random_recoil_flag = kwargs.pop('random_recoil', False)
        recoil_velocity = kwargs.pop('recoil_velocity', 0.01)
        max_scatter_probability = kwargs.pop('max_scatter_probability', 0.1)

        def dydt(t, y):
            return np.concatenate((
                self.drhodt(y[-3:], t, y[:-6]),
                recoil_velocity*self.force(y[-3:], t, y[:-6])*free_axes,
                y[-6:-3]))

        def random_recoil(t, y, dt):
            # Calculate the probability that all excited states can decay.
            # $P_i = Gamma_i dt n_i$, where $n_i$ is the population in state $i$
            # TODO: add in decay time (currently assumed to be one)
            P = np.zeros((self.hamiltonian.n-self.hamiltonian.ns[0],))
            for ii in range(self.hamiltonian.ns[0], self.hamiltonian.n):
                P[ii-self.hamiltonian.ns[0]] = y[self.density_index(ii, ii)]*dt
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

        if not random_recoil_flag:
            self.sol = solve_ivp(
                dydt, t_span, np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs)
        else:
            self.sol = solve_ivp_random(
                dydt, random_recoil, t_span,
                np.concatenate((self.rho0, self.v0, self.r0)),
                **kwargs
                )


    def observable(self, O, rho=None):
        """
        Observable returns the obervable O given density matrix rho.

        Parameters
        ----------
        O : array or array-like
            The matrix form of the observable operator.  Can have any shape,
            representing scalar, vector, or tensor operators, but the last two
            axes must correspond to the matrix of the operator and have the
            same dimensions of the generating Hamiltonian.  For example,
            a vector operator might have the shape (3, n, n), where n
            is the number of states and the first axis corresponds to x, y,
            and z.
        rho : [optional] array or array-like
            The density matrix.  The first two dimensions must have sizes
            (n, n), but there may be multiple instances of the density matrix
            tiled in the higher dimensions.  For example, a rho with (n, n, m)
            could have m instances of the density matrix at different times.

            If not specified, will get rho from the current solution stored in
            memory.

        Outputs
        -------
        observable: float or array
            observable has shape (O[:-2])+(rho[2:])
        """
        if rho is None:
            (t, r, v, rho) = self.reshape_sol()

        if rho.shape[:2]!=(self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('rho must have dimensions (n, n,...), where n '+
                             'corresponds to the number of states in the '+
                             'generating Hamiltonian. ' +
                             'Instead, shape of rho is %s.'%str(rho.shape))
        elif O.shape[-2:]!=(self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError('O must have dimensions (..., n, n), where n '+
                             'corresponds to the number of states in the '+
                             'generating Hamiltonian. ' +
                             'Instead, shape of O is %s.'%str(O.shape))
        else:
            avO = np.tensordot(O, rho, axes=[(-2, -1),(0, 1)])
            if np.allclose(np.imag(avO), 0):
                return np.real(avO)
            else:
                return avO


    def force(self, r, t, rho, return_details=False):
        if rho.shape[0] != self.hamiltonian.n:
            rho = self.reshape_rho(rho)

        f = np.zeros((3,) + rho.shape[2:])
        if return_details:
            f_laser_q = {}
            f_laser = {}

        for key in self.laserBeams:
            # First, determine the average mu_q:
            # This returns a (3,) + rho.shape[2:] array
            mu_q_av = self.observable(self.hamiltonian.d_q_bare[key], rho)

            if not return_details:
                delE = self.laserBeams[key].total_electric_field_gradient(np.real(r), t)

                for jj, q in enumerate(np.arange(-1., 2., 1.)):
                    f += np.real((-1)**q*mu_q_av[jj]*delE[:, 2-jj])
            else:
                f_laser_q[key] = np.zeros((3, 3, self.laserBeams[key].num_of_beams)
                                          + rho.shape[2:])
                f_laser[key] = np.zeros((3, self.laserBeams[key].num_of_beams)
                                        + rho.shape[2:])

                # Now, dot it into each laser beam:
                for ii, beam in enumerate(self.laserBeams[key].beam_vector):
                    if not self.transform_into_re_im:
                        delE = beam.electric_field_gradient(np.real(r), t)
                    else:
                        delE = beam.electric_field_gradient(r, t)

                    for jj, q in enumerate(np.arange(-1., 2., 1.)):
                        f_laser_q[key][:, jj, ii] += np.real((-1)**q*mu_q_av[jj]*delE[:, 2-jj])

                    f_laser[key][:, ii] = np.sum(f_laser_q[key][:, :, ii], axis=1)

                f+=np.sum(f_laser[key], axis=1)

        # Are we including magnetic forces?
        if self.include_mag_forces:
            # This function returns a matrix that (3, 3) with the format:
            # [dBx/dx, dBy/dx, dBz/dx; dBx/dy, dBy/dy, dBz/dy], and so on.
            # We need to dot, and su
            delB = self.magField.gradField(np.real(r))

            # What's the expectation value of mu?
            av_mu = self.observable(self.hamiltonian.mu, rho)

            # Now dot it into the gradient:
            f_mag = np.zeros(f.shape)
            for ii in range(3): # Loop over muxB_x, mu_yB_y, mu_zB_z
                f_mag += av_mu[ii]*delB[:, ii]

            # Add it into the regular force.
            f+=f_mag
        elif return_details:
            f_mag=np.zeros(f.shape)

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f


    def find_equilibrium_force(self, **kwargs):
        deltat = kwargs.pop('deltat', 500)
        itermax = kwargs.pop('itermax', 100)
        Npts = kwargs.pop('Npts', 5001)
        rel = kwargs.pop('rel', 1e-5)
        abs = kwargs.pop('abs', 1e-9)
        debug = kwargs.pop('debug', False)

        old_f_avg = np.array([np.inf, np.inf, np.inf])

        if debug:
            print('Finding equilbrium force at '+
                  'r=(%.2f, %.2f, %.2f) ' % (self.r0[0], self.r0[1], self.r0[2]) +
                  'v=(%.2f, %.2f, %.2f) ' % (self.v0[0], self.v0[1], self.v0[2]) +
                  'with deltat = %.2f, itermax = %d, Npts = %d, ' %  (deltat, itermax, Npts) +
                  'rel = %.1e and abs = %.1e' % (rel, abs)
                  )
            self.piecewise_sols = []

        ii=0
        while ii<itermax:
            if not Npts is None:
                kwargs['t_eval'] = np.linspace(ii*deltat, (ii+1)*deltat, int(Npts))

            self.evolve_density([ii*deltat, (ii+1)*deltat], **kwargs)
            (t, r, v, rho) = self.reshape_sol()
            f, f_laser, f_laser_q, f_mag = self.force(r, t, rho,
                                                      return_details=True)

            f_avg = np.mean(f, axis=1)

            if debug:
                print(ii, f_avg, np.sum(f_avg**2))
                self.piecewise_sols.append(self.sol)

            if (np.sum((old_f_avg-f_avg)**2)/np.sum((f_avg)**2)<rel or
                np.sum((old_f_avg-f_avg)**2)<abs):
                break;
            else:
                old_f_avg = copy.copy(f_avg)
                self.set_initial_rho(self.sol.y[:-6, -1])
                self.set_initial_position_and_velocity(self.sol.y[-3:, -1],
                                                       self.sol.y[-6:-3, -1])
                ii+=1

        f_mag = np.mean(f_mag, axis=1)

        f_laser_avg = {}
        f_laser_avg_q = {}
        for key in f_laser:
            f_laser_avg[key] = np.mean(f_laser[key], axis=2)
            f_laser_avg_q[key] = np.mean(f_laser_q[key], axis=3)

        Neq = np.real(np.diagonal(np.mean(rho, axis=2)))
        return (Neq, f_avg, f_laser_avg, f_laser_avg_q, f_mag, ii)


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

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

        it = np.nditer([R[0], R[1], R[2], V[0], V[1], V[2]],
                       flags=['refs_ok', 'multi_index'],
                        op_flags=[['readonly'], ['readonly'], ['readonly'],
                                  ['readonly'], ['readonly'], ['readonly']])

        if progress_bar:
            avgtime = 0.

        for (x, y, z, vx, vy, vz) in it:
            # Construct the rate equations:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            if progress_bar:
                tic = time.time()

            self.set_initial_position_and_velocity(r, v)
            if initial_rho is 'rateeq':
                self.set_initial_rho_from_rateeq()
            elif initial_rho is 'equally':
                self.set_initial_rho_equally()
            else:
                raise ValueError('Argument initial_rho=%s not understood'%initial_rho)

            if deltat_v is not None:
                vabs = np.sqrt(np.sum(v**2))
                if vabs==0.:
                    kwargs['deltat'] = deltat_tmax
                else:
                    kwargs['deltat'] = np.min([2*np.pi*deltat_v/vabs, deltat_tmax])

            if deltat_r is not None:
                rabs = np.sqrt(np.sum(r**2))
                if rabs==0.:
                    kwargs['deltat'] = deltat_tmax
                else:
                    kwargs['deltat'] = np.min([2*np.pi*deltat_r/rabs, deltat_tmax])

            Neq, F, F_laser, F_laser_q, F_mag, iterations = self.find_equilibrium_force(**kwargs)

            self.profile[name].store_data(it.multi_index, Neq, F, F_laser, F_mag,
                                          iterations, F_laser_q)

            if progress_bar:
                toc = time.time()

                avgtime = (it.iterindex*avgtime + (toc-tic))/(it.iterindex+1.0)

                printProgressBar(it.iterindex+1, it.itersize, prefix = 'Progress:',
                                 suffix = 'complete', decimals = 1, length = 40,
                                 remaining_time = (it.itersize-it.iterindex)*avgtime)

    def reshape_rho(self, rho):
        if self.transform_into_re_im:
            rho = rho.astype('complex128')

            if len(rho.shape) == 1:
                rho = self.U @ rho
            else:
                for jj in range(rho.shape[1]):
                    rho[:, jj] = self.U @ rho[:, jj]

        rho = rho.reshape((self.hamiltonian.n, self.hamiltonian.n) +
                          rho.shape[1:])

        """# If not:
        if self.transform_into_re_im:
            new_rho = np.zeros(rho.shape, dtype='complex128')
            for jj in range(new_rho.shape[2]):
                new_rho[:, :, jj] = (np.diag(np.diagonal(rho[:, :, jj])) +
                                     np.triu(rho[:, :, jj], k=1) +
                                     np.triu(rho[:, :, jj], k=1).T +
                                     1j*np.tril(rho[:, :, jj], k=-1) -
                                     1j*np.tril(rho[:, :, jj], k=-1).T)
            rho = new_rho"""

        return rho


    def reshape_sol(self):
        """
        Reshape the solution to have all the proper parts.
        """
        rho = self.reshape_rho(self.sol.y[:-6])

        return (self.sol.t, self.sol.y[-3:], self.sol.y[-6:-3], rho)
