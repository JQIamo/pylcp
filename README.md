Laser cooling physics tools in python.

Work in progress.

Currently, there is no setup.py.  In order to function, make sure to add the
directory containing pylcp into your path.  For example, if the path is
~/git/packages/pylcp, add a file to your site-packages directory in python
that contains ~/git/packages.  Then python will find pylcp.

Notes from DSB for windows:
The site-packages directory for a single user anaconda install should be
"C:\Users\<username>\AppData\Local\continuum\anaconda3\Lib\site-packages".
In that directory, add a file named, e.g., "custom_packages.pth", which
contains the absolute path to your github packages directory (~/git/packages
from above).
