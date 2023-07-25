pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX9 && mpirun.openmpi -np 9 test_cpuX9
popd
