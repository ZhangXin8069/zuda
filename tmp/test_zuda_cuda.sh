pushd /home/aistudio/work/zuda/tmp
nvcc -arch=sm_70 -o test_zuda_cuda /home/aistudio/work/zuda/test/test_zuda_cuda.cu -G -g && nvprof ./test_zuda_cuda
popd
