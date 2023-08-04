pushd /home/aistudio/work/pyquda_packages-main

source ./env.sh

pushd ${install_path}

# pip install -U numpy mpi4py cupy-cuda11x Cython typing_extensions -t ${install_path} #cupy-cuda11x:rely on `nvcc --version`,more in https://docs.cupy.dev/en/stable/install.html
# tar xzf ${file_path}/PyQuda-master.tar.gz && pushd PyQuda-master
pushd PyQuda-master
# cp ../quda-develop/build/lib/libquda.so ./
pip install -U . -t ${install_path}
python tests/test.dslash.qcu.py
popd
popd
# note
## change external-libraries/pyquda/utils/source.py and external-libraries/pyquda/field.py 
## as 
## from typing import List, Union
## from typing_extensions import Literal
## to debug


# #pragma once

# #ifdef __cplusplus
# extern "C" {
# #endif
# #include "/home/aistudio/work/zuda/include/zuda_cpu.h"

# typedef struct QcuParam_s {
#   int lattice_size[4];
# } QcuParam;

# void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param);
#   LatticeGauge U(*param.lattice_size[0], *param.lattice_size[1],
#                  *param.lattice_size[2], *param.lattice_size[3], 4, 3);
#   LatticeFermi b(*param.lattice_size[0], *param.lattice_size[1],
#                  *param.lattice_size[2], *param.lattice_size[3], 4, 3);
#   LatticeFermi x(*param.lattice_size[0], *param.lattice_size[1],
#                  *param.lattice_size[2], *param.lattice_size[3], 4, 3);
#   U.lattice_vec = gauge;
#   src.lattice_vec = fermion_in;
#   dslash(U, src, dest, false);
#   fermion_out = dest.fermion_out;
# #ifdef __cplusplus
# }
# #endif

popd
