pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX10 && mpirun.openmpi -np 10 test_zuda_cpuX10
popd
