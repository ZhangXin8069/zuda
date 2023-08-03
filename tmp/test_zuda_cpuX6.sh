pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX6 && mpirun.openmpi -np 6 test_zuda_cpuX6
popd
