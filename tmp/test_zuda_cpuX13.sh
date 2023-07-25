pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX13 && mpirun.openmpi -np 13 test_zuda_cpuX13
popd
