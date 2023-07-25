pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_cpuX.cc -o test_cpuX11 && mpirun.openmpi -np 11 test_cpuX11
popd
