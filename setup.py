#!/usr/bin/env python

from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='ImagesCameras',
    version='1.0.0',
    description='ToolBox for Image Processing based on Tensor',
    long_description=open('README.md').read(),
    author='Azorgz',
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('environment.yaml').readlines(),
    url='https://github.com/Azorgz/ImagesCameras.git',
    license='None',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)