pushd /home/aistudio/work/zuda/tmp
nvcc -O3 -arch=sm_70 --maxrregcount=255 -o test_zuda_cuda /home/aistudio/work/zuda/test/test_zuda_cuda.cu && nvprof ./test_zuda_cuda
popd
