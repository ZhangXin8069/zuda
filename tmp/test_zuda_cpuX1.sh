pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX1 && mpirun.openmpi -np 1 test_zuda_cpuX1
popd
