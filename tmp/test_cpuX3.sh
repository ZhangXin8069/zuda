pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX3 && mpirun.openmpi -np 3 test_cpuX3
popd
