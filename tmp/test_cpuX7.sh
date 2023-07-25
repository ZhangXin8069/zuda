pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX7 && mpirun.openmpi -np 7 test_cpuX7
popd
