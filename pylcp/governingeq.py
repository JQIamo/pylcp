import copy
import numpy as np
from .fields import magField as magFieldObject
from .fields import laserBeams as laserBeamsObject

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
        - A dictionary of laserBeams: if this is the case, the keys of the
          laser beams should address the
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
            self.laserBeams['g->e'] = copy.copy(args[0]) # Again, assume label is g->e
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not key is 'g->e':
                    raise KeyError('laserBeam dictionary should only contain ' +
                                   'a single key of \'g->e\' for the heutisticeq.')
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError('Key %s in dictionary lasersBeams ' % key +
                                     'is in not of type laserBeams.')
            self.laserBeams = copy.copy(laserBeams) # Now, assume that everything is the same.
        else:
            raise TypeError('laserBeams is not a valid type.')

        # Add in magnetic field:
        if callable(magField) or isinstance(magField, np.ndarray):
            self.magField = magField(magField)
        elif isinstance(magField, magFieldObject):
            self.magField = copy.copy(magField)
        else:
            raise TypeError('The magnetic field must be either a lambda ' +
                            'function or a magField object.')

        # Do argument checking specific
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

    def force(self):
        pass

    def generate_force_profile(self):
        pass

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
                    f = self.find_equilibrium_force()

                    F[jj] = f[axis]

                if np.diff(F)<0:
                    self.omega[axis] = np.sqrt(-np.diff(F)/(2*eps[axis]*mass))
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
