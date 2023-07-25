pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX12 && mpirun.openmpi -np 12 test_zuda_cpuX12
popd
