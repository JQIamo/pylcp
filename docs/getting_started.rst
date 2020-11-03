Getting Started
===============

`pylcp` has been designed to be as easy to use as possible and get the user
solving complicated problems in as few lines of code as possible.

The basic workflow for `pylcp` is to define the elements of the problem (the
laser beams, magnetic field, and Hamiltonian), combine these together in a
governing equation, and then calculate something of interest.

Creating the Hamiltonian
------------------------

The first step is define the Hamiltonian.   The full Hamiltonian is represented
as a series of blocks.  Each diagonal block represents a single state or a
group of states called a "manifold".  The diagonal blocks contain a field
independent part, :math:`H_0`, and a magnetic field dependent part
:math:`\mu_q`.  Off diagonal blocks connect different manifold together with
electric fields, and thus these correspond to dipole matrix elements between
the states :math:`d_q`.

Note that the Hamiltonian is constructed in the rotating frame, such that each
manifold's optical frequency is removed from the :math:`H_0` component.  As a
result, the manifolds are generally separated by optical frequencies.  However,
that need not be a requirement when constructing a Hamiltonian.

As a first example, let us consider a single ground state (labeled :math:`g`)
and an excited state (labeled :math:`e`) with some detuning :math:`\delta`::

  Hg = np.array([[0.]])
  He = np.array([[-delta]])
  mu_q = np.zeros((3, 1, 1))
  d_q = np.zeros((3, 1, 1))
  d_q[1, 0, 0] = 1.

Here we have defined the magnetic field independent part of the Hamiltonian
:math:`H_g` and :math:`H_e`, the magnetic field dependent part :math:`\mu_q`
(which is identically equal to zero) and the electric field :math:`d_q`
dependent part that drivers transitions between :math:`g` and :math:`e`.  The
:math:`\mu_q` and :math:`d_q` elements are represented in the spherical basis,
so they are actually vectors of matrcies, with the first element being the
:math:`q=-1`, the second :math:`q=0`, and the third :math:`q=+1`.  In our
example, the two-level system is magnetic field insensitive, and can only be
driven with :math:`\sigma^+` light.

There are shape requirements on the matrices and the arrays used to construct
the Hamiltonian.  Assume you create two manifolds, :math:`g` and :math:`e`,
with :math:`n` and :math:`m` states, respectively.  In this case, :math:`H_g`
must have shape :math:`n\times n` and the ground state :math:`\mu_q` must have
shape :math:`3\times n\times n`.  Likewise for the excited states.  The
:math:`d_q` matrix must have shape :math:`3\times n \times m`.

We then combine the whole thing together into the pylcp.hamiltonian class::

   hamiltonian = pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)

There are a host of functions for returning individual components of this
block Hamiltonian, documented in :doc:`hamiltonians`.

Laser beams
-----------
The next components is to define a collection of laser beams.  For example,
two create two counterpropagating laser beams ::

  laserBeams = pylcp.laserBeams([
          {'kvec':np.array([1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 'beta':beta},
          {'kvec':np.array([-1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 'beta':beta}
          ], beam_type=pylcp.infinitePlaneWaveBeam)

Here, we make the laser beam collection by passing a list of dictionaries, each
dictionary containing the keyword arguments to make individual
`pylcp.infinitePlaneWaveBeam` beams.  `kvec` specifies the k-vector of the laser,
`pol` specifies its polarization in the coordinate system specified by `pol_coord`,
`delta` specifies its frequency in the rotating frame (typically the detuning),
and `beta` specifies is saturation parameter.  The optioanl `beam_type` argument
specifies the subclass of pylcp.laserBeam to use in constructing the individual
laser beams.  More information can be found in :doc:`laser_fields`.


Magnetic field
--------------
The last component that one specifies the magnetic field.  For this example, we
will create a quadrupole magnetic field ::

  magField = pylcp.quadrupoleMagneticField(alpha)

Here, :math:`\alpha` is the strength of the magnetic field gradient.  There
are many types of magnetic fields to choose from, documented in
:doc:`magnetic_fields`.


Governing equation
------------------

Once all the components are created, we can combine them together into a
govening equation.  In this case, it is an optical Bloch equation ::

  obe = pylcp.obe(laserBeams, magField, hamiltonian)

And once you have your governing equation, you simply calculate the thing of
interest.  For example, if you wanted to calculate the force at locations :math:`R`
and velocities :math:`V`, you could use the `generate_force_profile` method ::

  obe.generate_force_profile(R, V)

All methods of the governing equations are documented in :doc:`governing_equations`.


Next steps
----------

Start looking at the :doc:`examples` for next steps; they contain a host of useful code
that can be easily borrowed to start a calculation.
