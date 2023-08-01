pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX9 && mpirun.openmpi -np 9 test_zuda_cpuX9
popd
