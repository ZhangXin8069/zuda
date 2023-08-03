# distutils: language = c++
# cython: language_level = 3

cimport numpy as np
from libcpp cimport bool


cdef extern from "../include/zuda_cpu.h":
    void ext_dslash(double * U_real, double * U_imag, double * src_real, double * src_imag, double * dest_real, double * dest_imag, int lat_x, int lat_y, int lat_z, int lat_t, int lat_s, int lat_c, bool test)
    void ext_cg(double * U_real, double * U_imag, double * b_real, double * b_imag, double * x_real, double * dx_imag, int lat_x, int lat_y, int lat_z, int lat_t, int lat_s, int lat_c, int MAX_ITER, double TOL, bool test)


def dslash_py(np.ndarray[np.double_t, ndim=1] U_real, np.ndarray[np.double_t, ndim=1] U_imag, np.ndarray[np.double_t, ndim=1] src_real, np.ndarray[np.double_t, ndim=1] src_imag, np.ndarray[np.double_t, ndim=1] dest_real, np.ndarray[np.double_t, ndim=1] dest_imag, int lat_x, int lat_y, int lat_z, int lat_t, int lat_s, int lat_c, bool test):
    ext_dslash(& U_real[0], & U_imag[0], & src_real[0], & src_imag[0], & dest_real[0], & dest_imag[0], lat_x, lat_y, lat_z, lat_t, lat_s, lat_c, test)


def cg_py(np.ndarray[np.double_t, ndim=1] U_real, np.ndarray[np.double_t, ndim=1] U_imag, np.ndarray[np.double_t, ndim=1] b_real, np.ndarray[np.double_t, ndim=1] b_imag, np.ndarray[np.double_t, ndim=1] x_real, np.ndarray[np.double_t, ndim=1] dx_imag, int lat_x, int lat_y, int lat_z, int lat_t, int lat_s, int lat_c, int MAX_ITER, double TOL, bool test):
    ext_cg( & U_real[0], & U_imag[0], & b_real[0], & b_imag[0], & x_real[0], & dx_imag[0], lat_x, lat_y, lat_z, lat_t, lat_s, lat_c, MAX_ITER, TOL, test)
