Getting Started
===============

`pylcp` has been designed to be as easy to use as possible and get the user
solving complicated problems in as few lines of code as possible.


Basic Usage
-----------
The basic workflow for `pylcp` is to define the elements of the problem (the
laser beams, magnetic field, and Hamiltonian), combine these together in a
governing equation, and then calculate something of interest.

The first step is define the problem.  The component of the problem is the
Hamiltonian.  The full Hamiltonian is represented as a series of blocks, and
so we first define all the blocks individually.  For this example, we assume
a ground state (labeled :math:`g`) and an excited state (labeled :math:`e`) with some
detuning :math:`\delta`::

  Hg = np.array([[0.]])
  He = np.array([[-delta]])
  mu_q = np.zeros((3, 1, 1))
  d_q = np.zeros((3, 1, 1))
  d_q[1, 0, 0] = 1.

  hamiltonian = pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)

Here we have defined the magnetic field independent part of the Hamiltonian :math:`H_g`
and :math:`H_e`, the magnetic field dependent part :math:`\mu_q`, and the electric field :math:`d_q`
dependent part that drivers transitions between :math:`g` and :math:`e`.

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
`pylcp.infinitePlaneWaveBeam` beams.

The last component that one specifies the magnetic field.  For this example, we
will create a quadrupole magnetic field ::
  magField = pylcp.quadrupoleMagneticField(alpha)
Here, :math:`\alpha` is the strength of the magnetic field gradient.

Once all the components are created, we can combine them together into a
govening equation.  In this case, it is an optical Bloch equation ::
  obe = pylcp.obe(laserBeams, magField, hamiltonian)

And once you have your governing equation, you simply calculate the thing of
interest.  For example, if you wanted to calculate the force at locations :math:`R`
and velocities :math:`V`, you could use the `generate_force_profile` method ::
  obe.generate_force_profile(R, V)

Examples
--------

There are plenty of examples contained in the `examples/` directory as Juypter
notebooks.  These are organized as follows:

- `examples/basics`: a series of examples of stationary atoms demonstrating
  the essential physics including laser broadening, Rabi flopping, damped Rabi
  flopping, optical pumping, adiabatic rapid passage, EIT, and STIRAP.
- `examples/molasses`: the most simple laser cooling process, optical molasses.
  These examples work up from the two-level molasses of textbooks all the way
  through to molasses on both type I and II
- `examples/MOTs`: a series of examples for magneto-optical traps, including
  calculating basic properties like trapping frequencies and damping rates,
  finding the capture velocity, simulating the temperature, adding in
  more complicated level structure, etc.
- `examples/bichromatic`: a simple example on calculating the bichromatic
  force.

There are also a few other examples that are mostly designed to test for proper
operation:

- `examples/basics/hamiltonians`: a series of examples of testing that the
  hamiltonian functions return the correct matrices.
- `examples/basics/lasers`: making sure laser beams have the correct profiles
- `examples/basics/magnetic_traps`: making sure atoms move roughly correctly
  inside magnetic traps.
