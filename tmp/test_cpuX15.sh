pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX15 && mpirun.openmpi -np 15 test_cpuX15
popd
