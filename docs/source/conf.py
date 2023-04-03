# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

# Specify the path where the file that need to be documented are located
sys.path.insert(0, '/home/fallog/Bureau/BIOSCIENCES/4A/S2/DEV_LOG/PROJET_DEV_LOG')
sys.path.insert(0, '/home/fallog/Bureau/BIOSCIENCES/4A/S2/DEV_LOG/PROJET_DEV_LOG/projet')

project = 'IdKit'
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
