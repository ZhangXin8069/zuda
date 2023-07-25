pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX1 && mpirun.openmpi -np 1 test_cpuX1
popd
