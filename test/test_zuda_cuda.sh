pushd /public/home/zhangxin/zuda/test
nvcc -O3 -arch=sm_70 -o test_zuda_cuda /public/home/zhangxin/zuda/test/test_zuda_cuda.cu && nvprof ./test_zuda_cuda
popd
