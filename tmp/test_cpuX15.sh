pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX15 && mpirun.openmpi -np 15 test_cpuX15
popd
