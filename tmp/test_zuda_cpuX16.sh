pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX16 && mpirun.openmpi -np 16 test_zuda_cpuX16
popd
