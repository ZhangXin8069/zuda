pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX13 && mpirun.openmpi -np 13 test_cpuX13
popd
