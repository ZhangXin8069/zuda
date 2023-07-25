pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX7 && mpirun.openmpi -np 7 test_cpuX7
popd
