import os
import sys

# Make the import works properly for different modules and submodules
# Adding current directory to the PYTHONPATH
sys.path.append(os.path.dirname(__file__))
# Adding directory above this one to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
