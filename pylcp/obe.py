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
from scipy.interpolate import interp1d
from .rateeq import rateeq
from .fields import laserBeams, magField
from .integration_tools import solve_ivp_random
from .common import (progressBar, random_vector, spherical_dot,
                     cart2spherical, spherical2cart, base_force_profile)
from .governingeq import governingeq

"""
The following is code that will be reintroduced after magnetic_forces branch is
pulled into master:

for key in self.laserBeams:
    k_ham = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k']
    for kvec in self.laserBeams[key].kvec():
        if not np.abs(np.linalg.norm(kvec)-k_ham)<1e-15:
            raise ValueError('Laser beam driving transition %s '%key +
                             'with wavevector k=%s '%str(kvec) +
                             'has different magnitude from that '+
                             'specified in the Hamiltonian, %s.'%str(k_ham))

"""
@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@numba.jit(nopython=True)
def dot(A, x):
    return A @ x

@numba.jit(nopython=True)
def dot_and_add(A, x, b):
    b += A @ x

def cartesian_vector_tensor_dot(a, B):
    if B.ndim == 2 and a.ndim == 1:
        # Single point:
        return np.dot(B, a)
    elif B.ndim == 2:
        # Constant B, variable a:
        return np.sum(a[np.newaxis, ...]*B[..., np.newaxis], axis=1)
    else:
        # Varaible a and variable B.  Will throw an error if a.shape[1:] != B.shape[2:]:
        return np.sum(a[np.newaxis, ...]*B[...], axis=1)


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


class obe(governingeq):
    """
    The optical Bloch equations

    This class constructs the optical Bloch equations from the given laser
    beams, magnetic field, and hamiltonian.

    Parameters
    ----------
    laserBeams : dictionary of pylcp.laserBeams, pylcp.laserBeams, or list of pylcp.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.
    magField : pylcp.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleraiton to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    transform_into_re_im : boolean
        Optional flag to transform the optical Bloch equations into real and
        imaginary components.  This helps to decrease computaiton time as it
        uses the symmetry :math:`\\rho_{ji}=\\rho_{ij}^*` to cut the number
        of equations nearly in half.  Default: True
    use_sparse_matrices : boolean or None
        Optional flag to use sparse matrices.  If none, it will use sparse
        matrices only if the number of internal states > 10, which would result
        in the evolution matrix for the density operators being a 100x100
        matrix.  At that size, there may be some speed up with sparse matrices.
        Default: None
    include_mag_forces : boolean
        Optional flag to inculde magnetic forces in the force calculation.
        Default: True
    r0 : array_like, shape (3,), optional
        Initial position.  Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity.  Default: [0., 0., 0.]
    """
    def __init__(self, laserBeams, magField, hamitlonian,
                 a=np.array([0., 0., 0.]), transform_into_re_im=True,
                 use_sparse_matrices=None, include_mag_forces=True,
                 r0=np.array([0., 0., 0.]), v0=np.array([0., 0., 0.])):

        super().__init__(**kwargs)

        # Add the Hamiltonian:
        self.hamiltonian = copy.copy(hamiltonian)
        self.hamiltonian.make_full_matrices()

        # Add lasers:
        self.laserBeams = {} # Laser beams are meant to be dictionary,
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeams(args[0])) # Assume label is g->e
        elif isinstance(laserBeams, laserBeams):
            self.laserBeams['g->e'] = copy.copy(args[0]) # Again, assume label is g->e
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeams):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams) # Now, assume that everything is the same.
        else:
            raise TypeError('laserBeams is not a valid type.')

        # Next, check to see if there is consistency in k:
        self.__check_consistency_in_lasers_and_d_q()

        # Add in magnetic field:
        if callable(magField) or isinstance(magField, np.ndarray):
            self.magField = magField(magField)
        elif isinstance(args[1], magField):
            self.magField = copy.copy(magField)
        else:
            raise TypeError('The magnetic field must be either a ' +
                            'function or a magField object.')

        # Add in constant acceleration:
        if not isinstance(args[3], np.ndarray):
            raise TypeError('Constant acceleration must be an numpy array.')
        elif args[3].size != 3:
            raise ValueError('Constant acceleration must have length 3.')
        else:
            self.constant_accel = args[3]

        self.transform_into_re_im = transform_into_re_im

        if use_sparse_matrices is None:
            if self.hamiltonian.n>10: # Generally offers a performance increase
                self.use_sparse_matrices = True
            else:
                self.use_sparse_matrices = False
        else:
            self.use_sparse_matrices = use_sparse_matrices

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol = None

        # There will be time-dependent and time-independent components of the optical
        # Bloch equations.  The time-independent parts are related to spontaneous
        # emission, applied magnetic field, and the zero-field Hamiltonian.  We
        # compute the latter-two directly from the commuatator.

        # Build the matricies that control evolution:
        self.ev_mat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        # If necessary, transform the evolution matrices:
        if self.transform_into_re_im:
            self.__transform_ev_matrices()

        if self.use_sparse_matrices:
            self.__convert_to_sparse()

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
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']
            self.ev_mat['d_q'][key] = [None]*3
            self.ev_mat['d_q*'][key] = [None]*3
            for q in range(3):
                self.ev_mat['d_q'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_bare[key][q]/4.
                )
                self.ev_mat['d_q*'][key][q] = self.__build_coherent_ev_submatrix(
                    gamma*self.hamiltonian.d_q_star[key][q]/4.
                )
            self.ev_mat['d_q'][key] = np.array(self.ev_mat['d_q'][key])
            self.ev_mat['d_q*'][key] = np.array(self.ev_mat['d_q*'][key])


    def __build_decay_ev(self):
        """
        This method constructs the decay portion of the OBE using the radiation
        reaction approximation.
        """
        d_q_bare = self.hamiltonian.d_q_bare
        d_q_star = self.hamiltonian.d_q_star

        self.decay_rates = {}
        self.decay_rates_truncated = {}
        self.decay_rho_indices = {}
        self.recoil_velocity = {}

        self.ev_mat['decay'] = np.zeros((self.hamiltonian.n**2,
                                         self.hamiltonian.n**2),
                                        dtype='complex128')

        # Go through each dipole moment and calculate:
        for key in d_q_bare:
            ev_mat = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2),
                               dtype='complex128')
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            # The first index we want to capture:
            for ii in range(self.hamiltonian.n):
                # The second index we want to capture:
                for jj in range(self.hamiltonian.n):
                    # The first sum index:
                    for kk in range(self.hamiltonian.n):
                        # The second sum index:
                        for ll in range(self.hamiltonian.n):
                            for mm, q in enumerate(np.arange(-1., 2., 1)):
                                # first term in the commutator, first part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, jj)] -= \
                                d_q_star[key][mm, ll, kk]*d_q_bare[key][mm, kk, ii]
                                # first term in the commutator, second part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(kk, ll)] += \
                                d_q_star[key][mm, kk, ii]*d_q_bare[key][mm, jj, ll]

                                # second term in the commutator, first part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ll, kk)] += \
                                d_q_star[key][mm, ll, ii]*d_q_bare[key][mm, jj, kk]
                                # second term in the commutator, second part:
                                ev_mat[self.__density_index(ii, jj),
                                       self.__density_index(ii, ll)] -= \
                                d_q_star[key][mm, jj, kk]*d_q_bare[key][mm, kk, ll]

            # Normalize:
            ev_mat = 0.5*gamma*ev_mat

            # Save the decay rates for the evolve_motion function:
            self.decay_rates[key] = -np.real(np.array(
                [ev_mat[self.__density_index(ii, ii), self.__density_index(ii, ii)]
                 for ii in range(self.hamiltonian.n)]
                ))

            # These are useful for the random evolution part:
            self.decay_rates_truncated[key] = self.decay_rates[key][self.decay_rates[key]>0]
            self.decay_rho_indices[key] = np.array([self.__density_index(ii, ii)
                for ii, rate in enumerate(self.decay_rates[key]) if rate>0]
            )
            self.recoil_velocity[key] = \
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['k']\
                /self.hamiltonian.mass

            self.ev_mat['decay'] += ev_mat

        return self.ev_mat['decay']


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


    def set_initial_rho(self, rho0):
        """
        Sets the initial :math:`\\rho` matrix

        Parameters:
        -----------
        rho0 : array_like
            The initial :math:`\\rho`.  It must have :math:`n^2` elements, where :math:`n`
            is the total number of states in the system.  If a flat array, it
            will be reshaped.
        """
        if np.any(np.isnan(rho0)) or np.any(np.isinf(rho0)):
            raise ValueError('rho0 has NaNs or Infs!')

        if rho0.size != self.hamiltonian.n**2:
            raise ValueError('rho0 should have n^2 elements.')

        if rho0.shape == (self.hamiltonian.n, self.hamiltonian.n):
            rho0 = rho0.flatten()

        if self.transform_into_re_im and rho0.dtype is np.dtype('complex128'):
            self.rho0 = self.Uinv @ rho0
        elif (not self.transform_into_re_im and
              not rho0.dtype is np.dtype('complex128')):
            self.rho0 = rho0.astype('complex128')
        else:
            self.rho0 = rho0

    def set_initial_rho_equally(self):
        """
        Sets the initial :math:`\\rho` matrix such that all states have the same
        population.
        """
        if self.transform_into_re_im:
            self.rho0 = np.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = np.zeros((self.hamiltonian.n**2,), dtype='complex128')

        for jj in range(self.hamiltonian.ns[0]):
            self.rho0[self.__density_index(jj, jj)] = 1/self.hamiltonian.ns[0]

    def set_initial_rho_from_populations(self, Npop):
        """
        Sets the diagonal elements of the initial :math:`\\rho` matrix

        Parameters
        ----------
        """
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
                        ev_mat -= np.real(Eq[ii])*self.ev_mat['reE'][key][ii]
                        ev_mat -= np.imag(Eq[ii])*self.ev_mat['imE'][key][ii]
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii in range(3):
                    if np.abs(Eq[ii])>1e-10:
                        ev_mat -= np.conjugate(Eq[ii])*self.ev_mat['d_q'][key][ii]
                        ev_mat -= Eq[ii]*self.ev_mat['d_q*'][key][ii]

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
                        drhodt -= ((-1.)**q*np.real(Eq[2-ii])*
                                   (self.ev_mat['reE'][key][ii] @ rho))
                        drhodt -= ((-1.)**q*np.imag(Eq[2-ii])*
                                   (self.ev_mat['imE'][key][ii] @ rho))
            else:
                Eq = self.laserBeams[key].total_electric_field(np.real(r), t)
                for ii, q in enumerate(np.arange(-1., 2., 1)):
                    if np.abs(Eq[2-ii])>1e-10:
                        drhodt -= ((-1.)**q*Eq[2-ii]*
                                   (self.ev_mat['d_q'][key][ii] @ rho))
                        drhodt -= ((-1.)**q*np.conjugate(Eq[2-ii])*
                                   (self.ev_mat['d_q*'][key][ii] @ rho))

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        for ii, q in enumerate(range(-1, 2)):
            if self.transform_into_re_im:
                if np.abs(B[ii])>1e-10:
                    drhodt -= self.ev_mat['B'][ii]*B[ii] @ rho
            else:
                Bq = cart2spherical(B)
                if np.abs(Bq[ii])>1e-10:
                    drhodt -= self.ev_mat['B'][ii]*np.conjugate(Bq[ii]) @ rho

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
        progress_bar = kwargs.pop('progress_bar', False)

        a = np.zeros((3,))

        if progress_bar:
            progress = progressBar()

        def dydt(t, y):
            if progress_bar and t<=t_span[1]:
                progress.update(t/t_span[1])

            return np.concatenate((self.drhodt(y[-3:], t, y[:-6]), a, y[-6:-3]))

        self.sol = solve_ivp(dydt, t_span,
                             np.concatenate((self.rho0, self.v0, self.r0)),
                             **kwargs)

        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        # Remake the solution:
        self.reshape_sol()

        return self.sol


    def evolve_motion(self, t_span, **kwargs):
        """
        This function evolves the optical bloch equations for some period of
        time, with all their potential glory!
        """
        free_axes = np.bitwise_not(kwargs.pop('freeze_axis', [False, False, False]))
        random_recoil_flag = kwargs.pop('random_recoil', False)
        max_scatter_probability = kwargs.pop('max_scatter_probability', 0.1)
        progress_bar = kwargs.pop('progress_bar', False)
        record_force = kwargs.pop('record_force', False)

        if progress_bar:
            progress = progressBar()

        if record_force:
            ts = []
            Fs = []

        def dydt(t, y):
            if progress_bar:
                progress.update(t/t_span[1])

            if record_force:
                F = self.force(y[-3:], t, y[:-6], return_details=True)

                ts.append(t)
                Fs.append(F)

                F = F[0]
            else:
                F = self.force(y[-3:], t, y[:-6], return_details=False)

            return np.concatenate((
                self.drhodt(y[-3:], t, y[:-6]),
                F*free_axes/self.hamiltonian.mass + self.constant_accel,
                y[-6:-3]
                ))

        def random_recoil(t, y, dt):
            num_of_scatters = 0
            total_P = 0.

            # Go over each block in the Hamiltonian and compute the decay:
            for key in self.decay_rates:
                P = dt*self.decay_rates_truncated[key]*np.real(y[self.decay_rho_indices[key]])

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

        if progress_bar:
            # Just in case the solve_ivp_random terminated due to an event.
            progress.update(1.)

        # Remake the solution:
        self.reshape_sol()

        # Interpolate the force:
        if record_force:
            f = interp1d(ts[:-1], np.array([f[0] for f in Fs[:-1]]).T)
            self.sol.F = f(self.sol.t)

            f = interp1d(ts[:-1], np.array([f[3] for f in Fs[:-1]]).T)
            self.sol.fmag = f(self.sol.t)

            self.sol.f = {}
            for key in Fs[0][1]:
                f = interp1d(ts[:-1], np.array([f[1][key] for f in Fs[:-1]]).T)
                self.sol.f[key] = f(self.sol.t)
                self.sol.f[key] = np.swapaxes(self.sol.f[key], 0, 1)

        return self.sol


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
            rho = self.sol.rho

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
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            if not return_details:
                delE = self.laserBeams[key].total_electric_field_gradient(np.real(r), t)

                # We are just looking at the d_q, whereas the full observable
                # is \nabla (d_q \cdot E^\dagger) + (d_q^* E)) =
                # 2 Re[\nabla (d_q\cdot E^\dagger)].  Putting in the units,
                # we see we need a factor of gamma/4, making
                # this 2 Re[\nabla (d_q\cdot E^\dagger)]/4 =
                # Re[\nabla (d_q\cdot E^\dagger)]/2
                for jj, q in enumerate(np.arange(-1., 2., 1.)):
                    f += np.real((-1)**q*gamma*mu_q_av[jj]*delE[:, 2-jj])/2
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
                        f_laser_q[key][:, jj, ii] += \
                        np.real((-1)**q*gamma*mu_q_av[jj]*delE[:, 2-jj])/2

                    f_laser[key][:, ii] = np.sum(f_laser_q[key][:, :, ii], axis=1)

                f+=np.sum(f_laser[key], axis=1)

        # Are we including magnetic forces?
        if self.include_mag_forces:
            # This function returns a matrix that is either (3, 3) (if constant)
            # or (3, 3, t.size).  The first two dimensions are like
            # [dBx/dx, dBy/dx, dBz/dx; dBx/dy, dBy/dy, dBz/dy], and so on.
            # We need to dot, and su
            delB = self.magField.gradField(np.real(r))

            # What's the expectation value of mu?  Returns (3,) or (3, t.size)
            av_mu = self.observable(self.hamiltonian.mu, rho)

            # Now dot it into the gradient:
            f_mag = cartesian_vector_tensor_dot(av_mu, delB)

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
        return_details = kwargs.pop('return_details', False)

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
        while True:
            if not Npts is None:
                kwargs['t_eval'] = np.linspace(ii*deltat, (ii+1)*deltat, int(Npts))

            self.evolve_density([ii*deltat, (ii+1)*deltat], **kwargs)
            f, f_laser, f_laser_q, f_mag = self.force(self.sol.r, self.sol.t, self.sol.rho,
                                                      return_details=True)

            f_avg = np.mean(f, axis=1)

            if debug:
                print(ii, f_avg, np.sum(f_avg**2))
                self.piecewise_sols.append(self.sol)

            if (np.sum((f_avg)**2)<abs or
                np.sum((old_f_avg-f_avg)**2)/np.sum((f_avg)**2)<rel or
                np.sum((old_f_avg-f_avg)**2)<abs):
                break;
            elif ii>=itermax-1:
                break;
            else:
                old_f_avg = copy.copy(f_avg)
                self.set_initial_rho(self.sol.rho[:, :, -1])
                self.set_initial_position_and_velocity(self.sol.r[:, -1],
                                                       self.sol.v[:, -1])
                ii+=1

        if return_details:
            f_mag = np.mean(f_mag, axis=1)

            f_laser_avg = {}
            f_laser_avg_q = {}
            for key in f_laser:
                f_laser_avg[key] = np.mean(f_laser[key], axis=2)
                f_laser_avg_q[key] = np.mean(f_laser_q[key], axis=3)

            Neq = np.real(np.diagonal(np.mean(self.sol.rho, axis=2)))
            return (f_avg, f_laser_avg, f_laser_avg_q, f_mag, Neq, ii)
        else:
            return f_avg


    def generate_force_profile(self, R, V,  **kwargs):
        """
        Method that maps out the equilbirium forces:
        """
        def default_deltat(r, v, deltat_v, deltat_r, deltat_tmax):
            deltat = None
            if deltat_v is not None:
                vabs = np.sqrt(np.sum(v**2))
                if vabs==0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2*np.pi*deltat_v/vabs, deltat_tmax])

            if deltat_r is not None:
                rabs = np.sqrt(np.sum(r**2))
                if rabs==0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2*np.pi*deltat_r/rabs, deltat_tmax])

            return deltat

        name = kwargs.pop('name', None)
        progress_bar = kwargs.pop('progress_bar', False)
        deltat_r = kwargs.pop('deltat_r', None)
        deltat_v = kwargs.pop('deltat_v', None)
        deltat_tmax = kwargs.pop('deltat_tmax', np.inf)
        deltat_func = kwargs.pop(
            'deltat_func',
            lambda r, v: default_deltat(r, v, deltat_v, deltat_r, deltat_tmax)
        )
        initial_rho = kwargs.pop('initial_rho', 'rateeq')

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

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

            if progress_bar:
                tic = time.time()

            self.set_initial_position_and_velocity(r, v)
            if initial_rho is 'rateeq':
                self.set_initial_rho_from_rateeq()
            elif initial_rho is 'equally':
                self.set_initial_rho_equally()
            else:
                raise ValueError('Argument initial_rho=%s not understood'%initial_rho)

            if not deltat_func(r, v) is None:
                kwargs['deltat'] = deltat_func(r, v)
            kwargs['return_details'] = True

            F, F_laser, F_laser_q, F_mag, Neq, iterations = self.find_equilibrium_force(**kwargs)

            self.profile[name].store_data(it.multi_index, Neq, F, F_laser, F_mag,
                                          iterations, F_laser_q)

            if progress_bar:
                progress.update((it.iterindex+1)/it.itersize)

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
        self.sol.rho = self.reshape_rho(self.sol.y[:-6])
        self.sol.r = np.real(self.sol.y[-3:])
        self.sol.v = np.real(self.sol.y[-6:-3])

        del self.sol.y
