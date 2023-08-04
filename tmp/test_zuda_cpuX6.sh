pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX6 && mpirun.openmpi -np 6 test_zuda_cpuX6
popd
