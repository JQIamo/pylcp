from setuptools import setup, find_packages

setup(
    name="pylcp",
    version="1.0.0",
    packages=find_packages(),
    #scripts=["say_hello.py"],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["docutils>=0.3",
                      "numpy>=1.18",
                      "numba>=0.48",
                      "scipy>=1.4.1"],
    python_requires=">=3.6, <4",
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst"],
        # Include example ipynb files:
        "examples":["*.ipynb"]
    },

    # metadata to display on PyPI
    author="Stephen Eckel, Daniel Barker, Eric Norrgard",
    author_email="pylcp@googlegroups.com",
    description="A package for calculating laser cooling physics",
    keywords="atomic physics, laser cooling, numerical integration",
    url="https://github.com/JQIamo/pylcp/",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/JQIamo/pylcp/issues",
        "Documentation": "https://python-laser-cooling-physics.readthedocs.io/en/stable",
        "Source Code": "https://github.com/JQIamo/pylcp/",
    },
    license="NIST",
    classifiers=[
        "Development Status :: 5 - Production/Stable"
        "Intended Audience :: Science/Research"
        "License :: Other/Proprietary License"
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
    ]

    # could also include long_description, download_url, etc.
)
