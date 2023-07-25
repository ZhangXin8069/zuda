pushd /home/aistudio/work/zuda/tmp
mpic++.openmpi /home/aistudio/work/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX5 && mpirun.openmpi -np 5 test_zuda_cpuX5
popd
