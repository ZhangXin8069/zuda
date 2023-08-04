from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext = Extension(name="test_zuda_cuda", 
                sources=["test_zuda_cuda.pyx", "test_zuda_cuda.cu"],
                extra_compile_args=["-arch=sm_70", "--maxrregcount=255","-O3"],
                include_dirs=[np.get_include()]) 

setup(
    version='0.0',
    ext_modules = cythonize(ext),
)