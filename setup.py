from setuptools import setup

setup(
    name='hyperbolic_ops',
    version='1.0',
    author='Seunghyuk Cho',
    author_email='shhj1998@postech.ac.kr',
    packages=['hyperbolic_ops'],
    install_requires=['torch>=1.12.0', 'numpy', 'geoopt']
)
