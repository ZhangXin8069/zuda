# init
_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo 'HOME:'${_HOME}
_NAME=$(basename "$0")
name='test'
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# source
## mkdir
mkdir ${_HOME}/bin -p
mkdir ${_HOME}/include -p 
mkdir ${_HOME}/lib -p 
mkdir ${_HOME}/scripts -p
mkdir ${_HOME}/test -p
mkdir ${_HOME}/tmp -p
mkdir ${_HOME}/build -p
mkdir ${_HOME}/doc -p

source ${_HOME}/tmp/scripts.sh

# do
## export
export CPATH=$CPATH:/usr/include/mpi/
# export PYTHONPATH=$(cd ~ && pwd)/external-libraries:$PYTHONPATH
# export LD_LIBRARY_PATH=$(cd ~ && pwd)/external-libraries/quda/build/lib/libquda.so:$LD_LIBRARY_PATH

## add alias

# done
