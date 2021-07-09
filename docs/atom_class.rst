Atom Class
==========

The atom class contains useful reference numbers for a given atomic species,
like mass, nuclear :math:`g`-factor, with states and transitions useful for
laser cooling.  Data comes from `Daniel Steck's "Alkali D-line data"
<https://steck.us/alkalidata/>`_, `Tobias Tiecke's "Properties of Potassium"
<http://www.tobiastiecke.nl/archive/PotassiumProperties.pdf>`_ and
`Michael Gehm's "Properties of 6Li"
<http://www.physics.ncsu.edu/jet/techdocs/pdf/PropertiesOfLi.pdf>`_.


Overview
--------

.. currentmodule:: pylcp

.. autosummary::

    atom


Detailed functions
------------------

.. autoclass:: pylcp.atom
  :members:

.. currentmodule:: pylcp.atom

.. autoclass:: pylcp.atom.state
  :members:

.. autoclass:: pylcp.atom.transition
  :members:
