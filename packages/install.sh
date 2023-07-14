source ./env.sh

mkdir ${install_path}
pip install -U pip numpy mpi4py cupy-cuda11x Cython typing_extensions -t ${install_path}

pushd ${install_path}

unzip -oq ${file_path}/quda-develop.zip
mv quda-develop quda
pushd quda
mkdir build
pushd build
cmake .. -DQUDA_DIRAC_DOMAIN_WALL=OFF -DQUDA_CLOVER_DYNAMIC=OFF -DQUDA_CLOVER_RECONSTRUCT=OFF -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF -DQUDA_DIRAC_TWISTED_CLOVER=OFF -DQUDA_DIRAC_TWISTED_MASS=OFF -DQUDA_INTERFACE_MILC=OFF -DQUDA_LAPLACE=ON -DQUDA_MULTIGRID=ON
cmake --build . -j160
cmake --install .
popd
popd

unzip -oq ${file_path}/PyQuda-master.zip
mv PyQuda-master PyQuda
pushd PyQuda
cp ../quda/build/lib/libquda.so ./
pip install . -t ${install_path}
popd

popd


