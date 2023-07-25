pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX4 && mpirun.openmpi -np 4 test_cpuX4
popd
