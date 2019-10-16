from setuptools import setup, find_packages

setup(
    name = 'pu-learn',
    version = '0.0.1',
    url = 'https://github.com/pillyshi/pu-learn',
    author = 'pillyshi',
    author_email = 'pillyshi21@gmail.com',
    description = 'Simple PU Learning',
    packages = ['pulearn'],
    install_requires = ['numpy', 'scipy', 'scikit-learn'],
)
