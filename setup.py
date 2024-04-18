from setuptools import setup
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

exec(open('planckpr4lensing/_version.py').read())
setup(name="planckpr4lensing",
      version=__version__,
      author=__author__,
      license='LGPL',
      description='External Cobaya likelihood package: Planck PR4(NPIPE) lensing',
      zip_safe=False,  # set to false if you want to easily access bundled package data files
      packages=['planckpr4lensing', 'planckpr4lensing.tests', 'planckpr4lensing.iswlens_jtliks'],
      package_data={'planckpr4lensing': ['*.yaml', '*.bibtex', 'data_pr4/*', 'data_pr4/**/*']},
      install_requires=['cobaya>=2.0.5'],
      test_suite='planckpr4lensing.tests',
      tests_require=['camb>=1.0.5']
      )
