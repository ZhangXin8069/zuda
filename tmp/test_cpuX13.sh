pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX13 && mpirun.openmpi -np 13 test_cpuX13
popd
