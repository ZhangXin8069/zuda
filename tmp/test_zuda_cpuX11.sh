pushd /home/zhangxin/zuda/tmp
mpic++.openmpi /home/zhangxin/zuda/test/test_zuda_cpuX.cc -o test_zuda_cpuX11 && mpirun.openmpi -np 11 test_zuda_cpuX11
popd
