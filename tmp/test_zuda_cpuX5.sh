pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX5 && mpirun.openmpi -np 5 test_zuda_cpuX5
popd
