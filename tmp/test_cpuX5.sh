pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX5 && mpirun.openmpi -np 5 test_cpuX5
popd
