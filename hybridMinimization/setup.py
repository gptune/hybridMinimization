from setuptools import setup
import os
import io

NAME = 'hybridMinimization'
DESCRIPTION = ''

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = dict()
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED=[]

EXTRAS = {
    'all': [
        'Sphinx>=1.8.2',
        'sphinx_rtd_theme',
        'numpy',
        'GPy',
        'scikit-learn', # Is version 0.22.1 needed here?  
        'scikit-optimize',
        'scipy',
        'dill',
        'termcolor',
    ],
}

setup(
  name=NAME,
  version=about['__version__'],
  license='BSD',
  author="Hengrui Luo",
  author_email='hrluo@lbl.gov',
  description=DESCRIPTION,
  packages=['hybridMinimization'],
  url='https://github.com/hrluo/hybridMinimization',
  install_requires=REQUIRED,
  extras_require=EXTRAS,
)