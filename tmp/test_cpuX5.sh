pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX5 && mpirun.openmpi -np 5 test_cpuX5
popd
