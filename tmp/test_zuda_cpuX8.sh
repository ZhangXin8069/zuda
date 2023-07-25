pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX8 && mpirun.openmpi -np 8 test_zuda_cpuX8
popd
