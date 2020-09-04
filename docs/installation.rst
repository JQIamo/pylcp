Installation instructions
=========================

Prerequisite: Python
--------------------

Install python packages.  `pylcp` has been test with python 3.6+.  It requires
numpy??


Recommended installation: via Python pip
----------------------------------------

This should happen soon, but it should be installable by `pip install pylcp`


Manual installation
-------------------
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
