# source
_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "PATH:"$_PATH
pushd ${_PATH}/../
source ./env.sh
popd

# pushd ${_PATH}/cpu
# make -j24
# make install 
# make clean
# popd

pushd ${_PATH}/cuda
make -j24
make install 
make clean
popd
