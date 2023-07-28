pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX4 && mpirun.openmpi -np 4 test_zuda_cpuX4
popd
