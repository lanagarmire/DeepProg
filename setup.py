from setuptools import setup, find_packages
import sys, os

VERSION = '0.0.1'

setup(name='garmire_survivalDL',
      version=VERSION,
      description="Deep-Learning framework for multi-omic and survival data integration",
      long_description="""""",
      classifiers=[],
      keywords='Deep-Learning multi-omics survival data integration',
      author='o_poirion',
      author_email='opoirion@hawaii.edu',
      url='',
      license='MIT',
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=[],
      )
