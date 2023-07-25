from lib import zuda_cpu
import os
import sys

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))

import numpy as np

from pyquda import init

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()


def applyDslash(Mp, p, U_seed):
    import cupy as cp
    from pyquda import core, quda
    from pyquda.enum_quda import QudaParity
    from pyquda.field import LatticeFermion
    from pyquda.utils import gauge_utils

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, U_seed)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = LatticeFermion(latt_size, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_size)

    # Dslash a = b
    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param,
                    QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param,
                    QudaParity.QUDA_ODD_PARITY)

    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico()


Lx, Ly, Lz, Lt = 16, 16, 16, 32
Nd, Ns, Nc = 4, 4, 3
latt_size = [Lx, Ly, Lz, Lt]
p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
p[0, 0, 0, 0, 0, 0] = 1
for x in range(Lx):
    for y in range(Ly):
        for z in range(Lz):
            for t in range(Lt):
                for s in range(Ns):
                    for c0 in range(Nc):
                        p[t, z, y, x, s, c0] = np.complex(np.random.randn(),np.random.randn())
Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
_Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
U = applyDslash(Mp, p, 5)
size = Lx*Ly*Lz*Lt*Ns*Nc
size_U = Lx*Ly*Lz*Lt*Ns*Nc*Nc
U_real = np.zeros(size_U, np.double)
U_imag = np.zeros(size_U, np.double)
p_real = np.zeros(size, np.double)
p_imag = np.zeros(size, np.double)
Mp_real = np.zeros(size, np.double)
Mp_imag = np.zeros(size, np.double)
print(U.shape)
print(Mp.shape)
for x in range(Lx):
    for y in range(Ly):
        for z in range(Lz):
            for t in range(Lt):
                for s in range(Ns):
                    for c0 in range(Nc):
                        index = x * Ly * Lz * Lt * Ns * Nc + y * Lz * Lt * Ns * \
                            Nc + z * Lt * Ns * Nc + t * Ns * Nc + s * Nc + c0
                        p_real[index] = p[t, z, y, x, s, c0].real
                        p_imag[index] = p[t, z, y, x, s, c0].imag
                        for c1 in range(Nc):
                            index_U = x * Ly * Lz * Lt * Ns * Nc * Nc + y * Lz * Lt * Ns * Nc * Nc + \
                                z * Lt * Ns * Nc * Nc + t * Ns * Nc * Nc + s * Nc * Nc + c0 * Nc + c1
                            U_real[index_U] = U[s, t, z, y, x,  c0, c1].real
                            U_imag[index_U] = U[s, t, z, y, x,
                                                c0, c1].imag  # what the fuck ?
MAX_ITER = 1e6
TOL = 1e-6
test = False
print("############ZUDA############")

zuda_cpu.dslash_py(U_real,
               U_imag,
               p_real,
               p_imag,
               Mp_real,
               Mp_imag,
               Lx,
               Ly,
               Lz,
               Lt,
               Ns,
               Nc,
               test)
for x in range(Lx):
    for y in range(Ly):
        for z in range(Lz):
            for t in range(Lt):
                for s in range(Ns):
                    for c0 in range(Nc):
                        index = x * Ly * Lz * Lt * Ns * Nc + y * Lz * Lt * Ns * \
                            Nc + z * Lt * Ns * Nc + t * Ns * Nc + s * Nc + c0
                        _Mp[t, z, y, x, s, c0] = complex(
                            Mp_real[index], Mp_imag[index])


# print("np.sum(Mp[0, 0, 0, 1])",np.sum(Mp[0, 0, 0, 1]))
# print("np.sum(Mp)",np.sum(Mp))
# print("np.sum(_Mp[0, 0, 0, 1])",np.sum(_Mp[0, 0, 0, 1]))
# print("np.sum(_Mp)",np.sum(_Mp))
reduce = Mp-_Mp
print("###reduce:", reduce[0, 0, 0, 1])
print("###np.linalg.norm(reduce):", np.linalg.norm(reduce))
# print(np.sum(_Mp[0, 0, 0, 1]))
# print(Mp[1, 0, 0, 1])
# print(_Mp[1, 0, 0, 1])
# print(Mp[0, 0, 0, 1])
# print(_Mp[0, 0, 0, 1])
