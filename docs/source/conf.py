# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

# Specify the path where the file that need to be documented are located
sys.path.insert(0, os.path.abspath(
    os.path.join('..', '..', 'projet')))
sys.path.insert(0, os.path.abspath(
    os.path.join('..', '..', 'projet', 'src')))

project = 'projet'
copyright = '2023'
author = """Simon Buré, Lionel Dalmau, Mayoran Raveendran, Olivia Seffacene, Jesus Uxue Mendez"""
release = '0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "recommonmark",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
