pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX14 && mpirun.openmpi -np 14 test_cpuX14
popd
