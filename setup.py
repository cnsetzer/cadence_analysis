from setuptools import setup
from numpy.distutils.core import setup, Extension
from setuptools import find_packages


if __name__ == "__main__":

    setup(name='cadence_analysis',
          version='0.0.1',
          description='Functions for processing results of a cadence simulation',
          url='http://github.com/cnsetzer/astrotog',
          author='Christian Setzer',
          author_email='christian.setzer@fysik.su.se',
          license='MIT',
          classifiers=[
              # How mature is this project? Common values are
              #   3 - Alpha
              #   4 - Beta
              #   5 - Production/Stable
              'Development Status :: 3 - Alpha',

              # Indicate who your project is intended for
              'Intended Audience :: Astronomers',

              # Pick your license as you wish (should match "license" above)
              'License :: OSI Approved :: MIT License',

              # Specify the Python versions you support here. In particular, ensure
              # that you indicate whether you support Python 2, Python 3 or both.
              'Programming Language :: Python :: 3.5',
          ],
          packages=find_packages(),
          ext_modules=[ext])
