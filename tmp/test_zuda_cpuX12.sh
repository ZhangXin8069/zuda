pushd /public/home/zhangxin/zuda/tmp
mpic++.openmpi /public/home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX12 && mpirun.openmpi -np 12 test_zuda_cpuX12
popd
