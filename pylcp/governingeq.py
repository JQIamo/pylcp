import copy
import numpy as np
from .fields import magField as magFieldObject
from .fields import laserBeams as laserBeamsObject
from scipy.optimize import root_scalar, root

class governingeq(object):
    """
    Governing equation base class

    This class is the basis for making all the governing equations in `pylcp`,
    including the rate equations, heuristic equation, and the optical Bloch
    equations.  Its methods are available to other governing equations.

    Parameters
    ----------
    laserBeams : dictionary of pylcp.laserBeams, pylcp.laserBeams, or list of pylcp.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations.  which transitions in the block diagonal hamiltonian.  It can
        be any of the following:

            * A dictionary of pylcp.laserBeams: if this is the case, the keys of
              the dictionary should match available :math:`d^{nm}` matrices
              in the pylcp.hamiltonian object.  The key structure should be
              `n->m`.
            * pylcp.laserBeams: a single set of laser beams is assumed to
              address the transition `g->e`.
            * a list of pylcp.laserBeam: automatically promoted to a
              pylcp.laserBeams object assumed to address the transtion `g->e`.

    magField : pylcp.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp.hamiltonian or None
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleraiton to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    r0 : array_like, shape (3,)
        Initial position.  Default: [0.,0.,0.]
    v0 : array_like, shape (3,)
        Initial velocity.  Default: [0.,0.,0.]
    """

    def __init__(self, laserBeams, magField, hamiltonian=None,
                 a=np.array([0., 0., 0.]), r0=np.array([0., 0., 0.]),
                 v0=np.array([0., 0., 0.])):
        self.set_initial_position_and_velocity(r0, v0)

        # Add lasers:
        self.laserBeams = {} # Laser beams are meant to be dictionary,
        if isinstance(laserBeams, list):
            self.laserBeams['g->e'] = copy.copy(laserBeamsObject(laserBeams)) # Assume label is g->e
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams['g->e'] = copy.copy(laserBeams) # Again, assume label is g->e
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams) # Now, assume that everything is the same.
        else:
            raise TypeError('laserBeams is not a valid type.')

        # Add in magnetic field:
        if callable(magField) or isinstance(magField, np.ndarray):
            self.magField = magFieldObject(magField)
        elif isinstance(magField, magFieldObject):
            self.magField = copy.copy(magField)
        else:
            raise TypeError('The magnetic field must be either a lambda ' +
                            'function or a magField object.')

        # Add the Hamiltonian:
        if hamiltonian is not None:
            self.hamiltonian = copy.copy(hamiltonian)
            self.hamiltonian.make_full_matrices()

            # Next, check to see if there is consistency in k:
            self.__check_consistency_in_lasers_and_d_q()

        # Check the acceleration:
        if not isinstance(a, np.ndarray):
            raise TypeError('Constant acceleration must be an numpy array.')
        elif a.size != 3:
            raise ValueError('Constant acceleration must have length 3.')
        else:
            self.constant_accel = a

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Set the initial sol to zero:
        self.sol = None

        # Set an attribute for the equillibrium position:
        self.r_eq = None


    def __check_consistency_in_lasers_and_d_q(self):
        # Check that laser beam keys and Hamiltonian keys match.
        for laser_key in self.laserBeams.keys():
            if not laser_key in self.hamiltonian.laser_keys.keys():
                raise ValueError('laserBeams dictionary keys %s ' % laser_key +
                                 'does not have a corresponding key the '+
                                 'Hamiltonian d_q.')


    def set_initial_position_and_velocity(self, r0, v0):
        """
        Sets the initial position and velocity

        Parameters
        ----------
        r0 : array_like, shape (3,)
            Initial position.  Default: [0.,0.,0.]
        v0 : array_like, shape (3,)
            Initial velocity.  Default: [0.,0.,0.]
        """
        self.set_initial_position(r0)
        self.set_initial_velocity(v0)

    def set_initial_position(self, r0):
        """
        Sets the initial position

        Parameters
        ----------
        r0 : array_like, shape (3,)
            Initial position.  Default: [0.,0.,0.]
        """
        self.r0 = r0
        self.sol = None

    def set_initial_velocity(self, v0):
        """
        Sets the initial velocity

        Parameters
        ----------
        v0 : array_like, shape (3,)
            Initial position.  Default: [0.,0.,0.]
        """
        self.v0 = v0
        self.sol = None

    def evolve_motion(self):
        pass

    def find_equilibrium_force(self):
        """
        Find the equilibrium force at the initial conditions

        Returns
        -------
        force : array_like
            Equilibrium force experienced by the atom
        """
        pass

    def force(self):
        """
        Find the instantaneous force

        Returns
        -------
        force : array_like
            Force experienced by the atom
        """
        pass

    def generate_force_profile(self):
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
        pass

    def find_equilibrium_position(self, axes, **kwargs):
        """
        Find the equilibrium position

        Uses the find_equilibrium force() method to calculate the where the
        :math:`\\mathbf{f}(\mathbf{r}, \mathbf{v}=0)=0`.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the trapping frequencies along.
            Here, :math:`\hat{x}` is index 0, :math:`\hat{y}` is index 1, and
            :math:`\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the trapping frquency along :math:`\hat{z}`.
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        r_eq : list or float
            The equilibrium positions along the selected axes.
        """
        if self.r_eq is None:
            self.r_eq = np.zeros((3,))

        def simple_wrapper(r_changing):
            r_wrap = self.r_eq.copy()
            r_wrap[axes] = r_changing

            self.set_initial_position_and_velocity(r_wrap, np.array([0.0, 0.0, 0.0]))
            F = self.find_equilibrium_force()

            return F[axes]

        #print('Initial guess: %s' % r_eqi[axes])
        if len(axes)>1:
            result = root(simple_wrapper, **kwargs)
            self.r_eq[axes] = result.x
        else:
            result = root_scalar(simple_wrapper, **kwargs)
            self.r_eq[axes] = result.root

        return self.r_eq

    def trapping_frequencies(self, axes, r=None, eps=0.01, **kwargs):
        """
        Find the trapping frequency

        Uses the find_equilibrium force() method to calculate the trapping
        frequency for the particular configuration.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the trapping frequencies along.
            Here, :math:`\hat{x}` is index 0, :math:`\hat{y}` is index 1, and
            :math:`\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the trapping frquency along :math:`\hat{z}`.
        r : array_like, optional
            The position at which to calculate the damping coefficient.  By
            default r=None, which defaults to calculating at the equilibrium
            position as found by the find_equilibrium_position() method.  If
            this method has not been run, it defaults to the origin.
        eps : float, optional
            The small numerical :math:`\epsilon` parameter used for calculating
            the :math:`df/dr` derivative.  Default: 0.01
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        omega : list or float
            The trapping frequencies along the selected axes.
        """
        self.omega = np.zeros(3,)

        if isinstance(eps, float):
            eps = np.array([eps]*3)

        if r is None and self.r_eq is None:
            r = np.array([0., 0., 0.])
        elif r is None:
            r = self.r_eq

        if hasattr(self, 'mass'):
            mass = self.mass
        else:
            mass = self.hamiltonian.mass

        for axis in axes:
            if not np.isnan(r[axis]):
                rpmdri = np.tile(r, (2,1)).T
                rpmdri[axis, 1] += eps[axis]
                rpmdri[axis, 0] -= eps[axis]

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(rpmdri[:, jj],
                                                           np.zeros((3,)))
                    f = self.find_equilibrium_force(**kwargs)

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.omega[axis] = np.sqrt(-np.diff(F)/(2*eps[axis]*mass))
                else:
                    self.omega[axis] = 0
            else:
                self.omega[axis] = 0

        return self.omega[axes]

    def damping_coeff(self, axes, r=None, eps=0.01, **kwargs):
        """
        Find the damping coefficent

        Uses the find_equilibrium force() method to calculate the damping
        coefficient for the particular configuration.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the damping coefficient(s) along.
            Here, :math:`\hat{x}` is index 0, :math:`\hat{y}` is index 1, and
            :math:`\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the damping parameter along :math:`\hat{z}`.
        r : array_like, optional
            The position at which to calculate the damping coefficient.  By
            default r=None, which defaults to calculating at the equilibrium
            position as found by the find_equilibrium_position() method.  If
            this method has not been run, it defaults to the origin.
        eps : float
            The small numerical :math:`\epsilon` parameter used for calculating
            the :math:`df/dv` derivative.  Default: 0.01
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        beta : list or float
            The damping coefficients along the selected axes.
        """
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
                    f = self.find_equilibrium_force(**kwargs)

                    F[jj] = f[axis]

                self.beta[axis] = -np.diff(F)/(2*eps[axis])
            else:
                self.beta[axis] = 0

        return self.beta[axes]
