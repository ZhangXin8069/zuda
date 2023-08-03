from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('test_zuda_cpu', ['test_zuda_cpu.pyx'], include_dirs=[np.get_include()])
]


setup(
    name='test_zuda_cpu',
    version='0.0',
    ext_modules=cythonize(extensions),
)
