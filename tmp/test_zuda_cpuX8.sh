pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX8 && mpirun.openmpi -np 8 test_zuda_cpuX8
popd
