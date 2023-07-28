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
echo "nvcc -o ${name} ${work_path}/${name}.cu && nvprof ./${name}" >>${name}.sh
echo "popd" >>${name}.sh
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd
