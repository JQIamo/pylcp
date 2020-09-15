# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pylcp'
copyright = '2020, Steve Eckel, Daniel Barker, Eric Norrgard, and others'
author = 'Steve Eckel, Daniel Barker, Eric Norrgard, and others'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#
# html_title = u'test vtest'

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = 'PyLCP'

# If true, “(C) Copyright …” is shown in the HTML footer. Default is True.
html_show_copyright = False

html_theme_options = {
    'description': 'A python package for simulating laser cooling physics',
    'logo': 'pylcp_logo.png',
    'font_size': 10,
    'caption_font_size': 8
}

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
# latex_documents = []

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = './_static/banner-large.png'

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
latex_use_parts = False

latex_elements = {
# The paper size ('letter' or 'a4').
'papersize': 'letter',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# necessary for unicode charactacters in pdf output
'inputenc': '',
'utf8extra': '',

# remove blank pages (between the title page and the TOC, etc.)
'classoptions': ',openany,oneside',
'babel' : '\\usepackage[english]{babel}',

# Additional stuff for the LaTeX preamble.
'preamble': r'''
  \usepackage{hyperref}
  \setcounter{tocdepth}{2}
'''
}

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True
#
