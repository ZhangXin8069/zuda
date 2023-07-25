pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_cpuX.cc -o test_cpuX11 && mpirun.openmpi -np 11 test_cpuX11
popd
