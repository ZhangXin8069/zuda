pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX6 && mpirun.openmpi -np 6 test_cpuX6
popd
