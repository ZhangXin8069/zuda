# source
_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "PATH:"$_PATH
pushd ${_PATH}/../
source ./env.sh
popd

# init
_NAME=$(basename "$0")
name='test_zuda_cpuX'
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# do
pushd ${tmp_path}
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
for i in $(seq 16); do
       echo "making ${name}${i}.sh in ${tmp_path}"
       echo "pushd ${tmp_path}" >${name}${i}.sh
       echo "mpic++.openmpi ${work_path}/${name}.cc -o ${name}${i} && mpirun.openmpi -np ${i} ${name}${i}" >>${name}${i}.sh
       echo "popd" >>${name}${i}.sh
done
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# init
_NAME=$(basename "$0")
name='test_zuda_cuda'
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# do
pushd ${tmp_path}
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
echo "making ${name}.sh in ${tmp_path}"
echo "pushd ${tmp_path}" >${name}.sh
# echo "nvcc -o ${name} ${work_path}/${name}.cu -g -G && nsys profile --stats=true ./${name}" >>${name}.sh
#echo "nvcc -arch=sm_70 -o ${name} ${work_path}/${name}.cu -G -g && nvprof ./${name}" >>${name}.sh
# echo "nvcc -O3 -arch=sm_70 -o ${name} ${work_path}/${name}.cu && nvprof ./${name}" >>${name}.sh
echo "nvcc -O3 -arch=sm_70 --maxrregcount=255 -o ${name} ${work_path}/${name}.cu && nvprof ./${name}" >>${name}.sh
# echo "nvcc -O3 -arch=sm_86 -o ./${name} ${work_path}/${name}.cu && ./${name} 2>&1 > log_${name}.txt" >>${name}.sh
# echo "nvcc -O3 -arch=sm_86 -o ${name} ${work_path}/${name}.cu && nsys profile --stats=true -o ./${name}.qdrep ./${name}  && nsys export -o ./${name}.txt -text ./${name}.qdrep" >>${name}.sh
echo "popd" >>${name}.sh
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# # init
# _NAME=$(basename "$0")
# name="test_zuda_cpu"
# work_name="test"
# tmp_name="tmp"
# build_name="build"
# work_path=${_HOME}/${work_name}
# tmp_path=${_HOME}/${tmp_name}
# build_path=${_HOME}/${build_name}

# # do
# pushd ${tmp_path}
# echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
# echo "making ${name}.sh in ${tmp_path}"
# echo "pushd ${build_path}" >${name}.sh
# echo "rm -rf ./build
# python ./setup_${name}.py build_ext --inplace
# for i in \$(find ./ -type f -name *.so); do
#        mv \${i} ${_HOME}/lib
#        echo "mv \${i} ${_HOME}/lib"
# done
# rm -rf ./build
# popd
# pushd ${work_path}
# rm -rf ./lib
# ln -s ../lib/ .
# python ./${name}.py
# popd
# " >>${name}.sh
# echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
# popd

# init
_NAME=$(basename "$0")
name="test_zuda_cpu"
tmp_name="tmp"
work_path="/home/aistudio/work/pyquda_packages-main"
tmp_path=${_HOME}/${tmp_name}

# do
pushd ${tmp_path}
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
echo "pushd ${work_path}" >${name}.sh
echo '
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
' >>${name}.sh
echo "popd" >>${name}.sh
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# done
