from setuptools import setup
from setuptools.extension import Extension


setup(
    name = "msaexp_olf",
    author = "Vasily Kokorev",
    author_email = "vasily.kokorev.astro@gmail.com",
    description = "Optimal Line Fitting with msaexp",
    version = "0.0.1",
    license = "MIT",
    url = "https://github.com/vasilykokorev/XXX",  
    packages=['olf'],
    scripts=[],
    install_requires = ['numpy',
                        'scipy',
                        'matplotlib',
                        'astropy',
                        'multiprocess',
                        'tqdm',
                        'msaexp',
                        'grizli',
                        'eazy',
                        'numba'],
)