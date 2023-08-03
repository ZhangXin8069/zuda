pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX15 && mpirun.openmpi -np 15 test_zuda_cpuX15
popd
