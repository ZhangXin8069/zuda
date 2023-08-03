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
echo "nvcc -O3 -arch=sm_86 -o ${name} ${work_path}/${name}.cu && nvprof ./${name}" >>${name}.sh
# echo "nvcc -O3 -arch=sm_86 -o ./${name} ${work_path}/${name}.cu && ./${name} 2>&1 > log_${name}.txt" >>${name}.sh
# echo "nvcc -O3 -arch=sm_86 -o ${name} ${work_path}/${name}.cu && nsys profile --stats=true -o ./${name}.qdrep ./${name}  && nsys export -o ./${name}.txt -text ./${name}.qdrep" >>${name}.sh
echo "popd" >>${name}.sh
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# init
_NAME=$(basename "$0")
name="test_zuda_cpu"
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# do
pushd ${tmp_path}
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
echo "making ${name}.sh in ${tmp_path}"
echo "pushd ${tmp_path}" >${name}.sh
echo "rm -rf ./build
python ./setup_${name}.py build_ext --inplace
for i in $(find ./ -type f -name "*.so"); do
       mv ${i} ${_HOME}/lib
       echo "mv ${i} ${_HOME}/lib"
done
rm -rf ./build
pushd ${work_path}
rm -rf lib
ln -s ../lib/ .
python ./test_${name}.py
popd
" >>${name}.sh
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# done
