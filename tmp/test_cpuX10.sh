pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX10 && mpirun.openmpi -np 10 test_cpuX10
popd
