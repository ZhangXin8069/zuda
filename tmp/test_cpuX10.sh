pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX10 && mpirun.openmpi -np 10 test_cpuX10
popd
