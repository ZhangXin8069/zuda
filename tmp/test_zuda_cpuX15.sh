pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX15 && mpirun.openmpi -np 15 test_zuda_cpuX15
popd
