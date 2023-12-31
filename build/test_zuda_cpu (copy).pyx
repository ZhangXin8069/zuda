# distutils: language = c++
# cython: language_level = 3

cimport numpy
import numpy
from libcpp cimport bool


cdef extern from "../include/zuda_cpu.h":
    ctypedef struct ext_QcuParam:
        int lattice_size[4]
    void ext_dslashQcu(void *fermion_out,const void *fermion_in,const void *gauge,const ext_QcuParam *param)

def dslashQcu(numpy.ndarray fermion_out, numpy.ndarray fermion_in, numpy.ndarray gauge, QcuParam param):
    cdef size_t ptr_uint64
    ptr_uint64 = fermion_out.ctypes.data
    cdef void *fermion_out_ptr = <void *>ptr_uint64
    ptr_uint64 = fermion_in.ctypes.data
    cdef void *fermion_in_ptr = <void *>ptr_uint64
    ptr_uint64 = gauge.ctypes.data
    cdef void *gauge_ptr = <void *>ptr_uint64
    ext_dslashQcu(fermion_out_ptr, fermion_in_ptr, gauge_ptr, &param.param)

class QcuParam:
    ext_QcuParam param
    def __init__(self):
        pass

    @property
    def lattice_size(self):
        return self.param.lattice_size

    @lattice_size.setter
    def lattice_size(self, value):
        self.param.lattice_size = value