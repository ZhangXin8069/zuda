from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('zuda', ['zuda.pyx'], include_dirs=[np.get_include()])
]


setup(
    name='zuda',
    version='0.0',
    ext_modules=cythonize(extensions),
)