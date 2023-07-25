pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX8 && mpirun.openmpi -np 8 test_cpuX8
popd
