pylcp
=================================

[![Documentation Status](https://readthedocs.org/projects/python-laser-cooling-physics/badge/?version=latest)](https://python-laser-cooling-physics.readthedocs.io/en/latest/?badge=latest) [![GitHub version](https://badge.fury.io/gh/jqiAMO%2Fpylcp.svg)](https://badge.fury.io/gh/jqiAMO%2Fpylcp) [![PyPI version](https://badge.fury.io/py/pylcp.svg)](https://badge.fury.io/py/pylcp) [![Google group : SSFAM News](https://img.shields.io/badge/Google%20Group-pylcp-blue.svg)](https://groups.google.com/g/pylcp)

`pylcp` is a python package meant to help with the calculation of a variety of
interesting quantities in laser cooling physics.
It allows automatic generation of the optical Bloch equations (or some approximation
thereof) given an atom's or molecule's internal Hamiltonian, a set of laser beams, and
a magnetic field.

Installation
------------
You can install on the command line using pip:
```
  pip install pylcp
```
You can also manually check out the package from GitHub, navigate to the
directory, and use:
```
  python setup.py install
```
If you wish to participate in development, you should use:
```
  python setup.py develop
```
which does the standard thing and puts an entry for `pylcp` in your
`easy_path.pth` in your python installation.

Basic Usage
-----------
The basic workflow for `pylcp` is to define the elements of the problem (the
laser beams, magnetic field, and Hamiltonian), combine these together in a
governing equation, and then calculate something of interest.

The first step is to define the problem.
The component of the problem is the Hamiltonian.
The full Hamiltonian is represented as a series of blocks, and so we first
define all the blocks individually.
For this example, we assume a ground state (labeled `g`) and an excited state
(labeled `e`) with some detuning `delta`:
```
  Hg = np.array([[0.]])
  He = np.array([[-delta]])
  mu_q = np.zeros((3, 1, 1))
  d_q = np.zeros((3, 1, 1))
  d_q[1, 0, 0] = 1.

  hamiltonian = pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)
```

We have defined the magnetic field independent part of the Hamiltonian `Hg`
and `He`, the magnetic field dependent part `mu_q`, and the electric field
dependent part `d_q` that drives transitions between `g` and `e`.

The next component is a collection of laser beams.
For this example, we create two counterpropagating laser beams:
```
  laserBeams = pylcp.laserBeams([
          {'kvec':np.array([1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 's':norm_intensity},
          {'kvec':np.array([-1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 's':norm_intensity}
          ], beam_type=pylcp.infinitePlaneWaveBeam)
```

We make the laser beam collection by passing a list of dictionaries, each
dictionary containing the keyword arguments to make individual
`pylcp.infinitePlaneWaveBeam` laser beams.

The last component is the magnetic field.
For this example, we will create a quadrupole magnetic field:
```
  magField = pylcp.quadrupoleMagneticField(alpha)
```

Once all the components are created, we can combine them together into a
govening equation.
In this case, it is an optical Bloch equation:
```
  obe = pylcp.obe(laserBeams, magField, hamiltonian)
```

Once you have your governing equation, you can calculate quantities of
interest.
For example, if you wanted to calculate the force at locations `R`
and velocities `V`, you could use the `generate_force_profile` method:
```
  obe.generate_force_profile(R, V)
```

There are plenty of examples contained in the `examples/` directory as Juypter
notebooks.

Further documentation is available at https://python-laser-cooling-physics.readthedocs.io.
