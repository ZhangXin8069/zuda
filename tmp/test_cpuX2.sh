pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX2 && mpirun.openmpi -np 2 test_cpuX2
popd
