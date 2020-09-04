Welcome to pylcp!
=================================

`pylcp` is a python package meant to help with the calculation of a variety of
interesting quantities in laser cooling physics.  At its heart, it allows for
automatic generation of the optical Bloch equations or some approximation
thereof given a atom or molecule internal Hamiltonian, a set of laser beams, and
a possible magnetic field.

Installation
------------
In the future, this should be as simple as `pip install pylcp`

But currently, there is no setup.py.  In order to function, make sure to add the
directory containing pylcp into your path.  For example, if the path is
~/git/packages/pylcp, add a file to your site-packages directory in python
that contains ~/git/packages.  Then python will find pylcp.

Notes from DSB for windows:
The site-packages directory for a single user anaconda install should be
"C:\Users\<username>\AppData\Local\continuum\anaconda3\Lib\site-packages".
In that directory, add a file named, e.g., "custom_packages.pth", which
contains the absolute path to your github packages directory (~/git/packages
from above).


Basic Usage
-----------
The basic workflow for `pylcp` is to define the elements of the problem (the
laser beams, magnetic field, and Hamiltonian), combine these together in a
governing equation, and then calculate something of interest.

The first step is define the problem.  The component of the problem is the
Hamiltonian.  The full Hamiltonian is represented as a series of blocks, and
so we first define all the blocks individually.  For this example, we assume
a ground state (labeled `g`) and an excited state (labeled `e`) with some
detuning `delta`:
```
  Hg = np.array([[0.]])
  He = np.array([[-delta]])
  mu_q = np.zeros((3, 1, 1))
  d_q = np.zeros((3, 1, 1))
  d_q[1, 0, 0] = 1.

  hamiltonian = pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)
```
Here we have defined the magnetic field independent part of the Hamiltonian `Hg`
and `He`, the magnetic field dependent part `mu_q`, and the electric field `d_q`
dependent part that drivers transitions between `g` and `e`.

The next components is to define a collection of laser beams.  For example,
two create two counterpropagating laser beams

```
  laserBeams = pylcp.laserBeams([
          {'kvec':np.array([1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 'beta':beta},
          {'kvec':np.array([-1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
           'pol_coord':'spherical', 'delta':delta, 'beta':beta}
          ], beam_type=pylcp.infinitePlaneWaveBeam)
```

Here, we make the laser beam collectiov by passing a list of dictionaries, each
dictionary containing the keyword arguments to make individual
`pylcp.infinitePlaneWaveBeam` beams.

The last component that one specifies the magnetic field.  For this example, we
will create a quadrupole magnetic field:
```
  magField = pylcp.quadrupoleMagneticField(alpha)
```

Once all the components are created, we can combine them together into a
govening equation.  In this case, it is an optical Bloch equation:
```
  obe = pylcp.obe(laserBeams, magField, hamiltonian)
```

And once you have your governing equation, you simply calculate the thing of
interest.  For example, if you wanted to calculate the force at locations `R`
and velocities `V`, you could use the `generate_force_profile` method:
```
  obe.generate_force_profile(R, V)
```

There are plenty of examples contained in the `examples/` directory as Juypter
notebooks.

Further documentation is available in the full API.
