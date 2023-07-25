pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX8 && mpirun.openmpi -np 8 test_cpuX8
popd
