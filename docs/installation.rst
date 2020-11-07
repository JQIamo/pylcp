Installation instructions
=========================

Prerequisites
-------------

Install python packages and packages for scientific computing in python.
Specifically, `pylcp` uses `numpy`, `scipy`, `numba`.  We recommend installing
python and the supporting packages via the Anaconda distribution; `pylcp` has
been tested and found to work with Anaconda versions 2020.02+ (python 3.7).

Recommended installation: via Python pip
----------------------------------------

Install via pip::

  pip install pylcp

This automatically install `pylcp` into your python installation.  Please report
issues to the GitHub page if you have any problems.

Manual installation
-------------------

One can also manually check out the package from GitHub, navigate to the
directory, and use::

  python setup.py install

If one wishes to participate in development, one should use::

  python setup.py develop

which does the standard thing and puts an entry for pylcp in your easy_path.pth
in your python installation.
