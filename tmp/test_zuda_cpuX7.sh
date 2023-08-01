pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX7 && mpirun.openmpi -np 7 test_zuda_cpuX7
popd
