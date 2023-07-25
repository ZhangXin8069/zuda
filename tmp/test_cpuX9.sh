pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX9 && mpirun.openmpi -np 9 test_cpuX9
popd
