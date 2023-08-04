pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX2 && mpirun.openmpi -np 2 test_zuda_cpuX2
popd
