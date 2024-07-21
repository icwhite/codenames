import os

from setuptools import find_packages, setup


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f if line.strip() != '']


setup(
    name='codenames',
    version='1.0.0',
    description='Code for Cross-Cultural Communication with RSA in Codenames',
    url='https://github.com/',
    author='Michelle Pan, Sashrika Pandey, and Isadora White',
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt'),
    license='LICENCE',
)