pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX4 && mpirun.openmpi -np 4 test_zuda_cpuX4
popd
