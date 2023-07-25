pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX16 && mpirun.openmpi -np 16 test_cpuX16
popd
