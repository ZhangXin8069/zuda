pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX12 && mpirun.openmpi -np 12 test_cpuX12
popd
