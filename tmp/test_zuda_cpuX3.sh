pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX3 && mpirun.openmpi -np 3 test_zuda_cpuX3
popd
