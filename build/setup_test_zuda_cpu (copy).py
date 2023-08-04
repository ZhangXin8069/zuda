from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext = Extension(name="test_zuda_cpu", 
                sources=["test_zuda_cpu.pyx"],
                extra_compile_args=["-O3"],
                include_dirs=[np.get_include()]) 
setup(
    version='0.0',
    ext_modules = cythonize(ext),
)
