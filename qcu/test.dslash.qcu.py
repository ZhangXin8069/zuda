from pyquda.utils import gauge_utils
from pyquda.field import LatticeFermion
from pyquda.enum_quda import QudaParity
from pyquda import core, quda
import cupy as cp
import os
import sys

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))

import numpy as np

from pyquda import init, pyqcu

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

Lx, Ly, Lz, Lt = 32, 32, 32, 64
Nd, Ns, Nc = 4, 4, 3
latt_size = [Lx, Ly, Lz, Lt]

U_seed = 0
p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
p[0, 0, 0, 0, 0, 0] = 1
p[0, 1, 0, 1] = 3

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

Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
Mp[:] = b.lexico()
print(Mp[0, 0, 0, 1])
print(np.linalg.norm(Mp))

b = LatticeFermion(latt_size)
# Dslash a = b
param = pyqcu.QcuParam()
param.lattice_size = latt_size
pyqcu.dslashQcu(b.even_ptr, a.odd_ptr, U.data_ptr, param, 0)
pyqcu.dslashQcu(b.odd_ptr, a.even_ptr, U.data_ptr, param, 1)
Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
Mp1[:] = b.lexico()
print(Mp1[0, 0, 0, 1])
print(np.linalg.norm(Mp1))


print(np.linalg.norm(Mp - Mp1))
