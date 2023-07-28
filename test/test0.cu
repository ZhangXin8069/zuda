#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include<time.h>

// // Define dimensions
// const int LAT_X = 32;
// const int LAT_Y = 32;
// const int LAT_Z = 32;
// const int LAT_T = 32;
// const int NUM_PARITIES = 2;
// const int VOLUME = LAT_X * LAT_Y * LAT_Z * LAT_T;
// // Complex number structure
// struct Complex
// {
//     double real;
//     double imag;

//     __device__ Complex operator*(const Complex &other) const
//     {
//         Complex result;
//         result.real = real * other.real - imag * other.imag;
//         result.imag = real * other.imag + imag * other.real;
//         return result;
//     }

//     __device__ Complex operator+(const Complex &other) const
//     {
//         Complex result;
//         result.real = real + other.real;
//         result.imag = imag + other.imag;
//         return result;
//     }

//     __device__ Complex operator-(const Complex &other) const
//     {
//         Complex result;
//         result.real = real - other.real;
//         result.imag = imag - other.imag;
//         return result;
//     }
// };

// __device__ Complex conj(const Complex &c)
// {
//     Complex result;
//     result.real = c.real;
//     result.imag = -c.imag;
//     return result;
// }

// // Fermi field class
// class FermiField
// {
// private:
//     Complex *field;

// public:
//     int numParities;
//     __host__ __device__ FermiField(Complex *fieldPtr, int parities) : field(fieldPtr), numParities(parities) {}

//     __host__ __device__ Complex &getField(int index, int parity)
//     {
//         return field[parity * VOLUME + index];
//     }

//     __host__ __device__ void setField(int index, int parity, const Complex &value)
//     {
//         field[parity * VOLUME + index] = value;
//     }
// };

// // Gauge field class
// class GaugeField
// {
// private:
//     Complex *field;
//     int numParities;

// public:
//     __host__ __device__ GaugeField(Complex *fieldPtr, int parities) : field(fieldPtr), numParities(parities) {}

//     __device__ Complex &getLink(int index, int mu, int parity)
//     {
//         return field[mu * (numParities * VOLUME) + parity * VOLUME + index];
//     }

//     __device__ void setLink(int index, int mu, int parity, const Complex &value)
//     {
//         field[mu * (numParities * VOLUME) + parity * VOLUME + index] = value;
//     }
// };

// __global__ void setupRandomGenerator(curandState *devStates)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < VOLUME)
//     {
//         curand_init(1234, idx, 0, &devStates[idx]);
//     }
// }

// __global__ void initInputFields(Complex *devFermiField, Complex *devGaugeField, curandState *devStates)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < VOLUME)
//     {
//         int parity = idx % NUM_PARITIES;

//         curandState localState = devStates[idx];

//         devFermiField[parity * VOLUME + idx] = {curand_uniform(&localState), curand_uniform(&localState)};

//         for (int dir = 0; dir < 4; dir++)
//         {
//             devGaugeField[(dir * NUM_PARITIES + parity) * VOLUME + idx] = {curand_uniform(&localState), curand_uniform(&localState)};
//         }

//         devStates[idx] = localState;
//     }
// }

// // Kernel for the Dslash4 operation
// __global__ void Dslash4(FermiField fermiField, GaugeField gaugeField, FermiField resultField, int parity)
// {
//     bool dag = true;
//     const double a = 2.0;
//     const double mass = 1.0;
//     Complex i = {0.0, 1};
//     Complex Half = {0.5, 0.0};
//     Complex flag = {(dag == true) ? -1.0 : 1.0, 0};
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < VOLUME)
//     {
//         int x = idx % LAT_X;
//         int y = (idx / LAT_X) % LAT_Y;
//         int z = (idx / (LAT_X * LAT_Y)) % LAT_Z;
//         int t = (idx / (LAT_X * LAT_Y * LAT_Z)) % LAT_T;

//         // mass term
//         for (int s = 0; s < fermiField.numParities; s++)
//         {
//             Complex value = fermiField.getField(idx, s);
//             resultField.setField(idx, s, value *Complex{-(a + mass), 0.0});
//         }

//         // backward x
//         int b_x = (x + LAT_X - 1) % LAT_X;
//         Complex tmp = (fermiField.getField(idx, 0) + flag * fermiField.getField(idx, 1)) * Half * gaugeField.getLink(idx, 0, parity);
//         resultField.setField(b_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 0, resultField.getField(b_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(b_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 1, resultField.getField(b_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 1) + flag * tmp);

//         // forward x
//         int f_x = (x + 1) % LAT_X;
//         tmp = (fermiField.getField(idx, 0) - flag * fermiField.getField(idx, 1)) * Half * conj(gaugeField.getLink(f_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 0, parity));
//         resultField.setField(f_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 0, resultField.getField(f_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(f_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 1, resultField.getField(f_x + LAT_X * (y + LAT_Y * (z + LAT_Z * t)), 1) - flag * tmp);

//         // backward y
//         int b_y = (y + LAT_Y - 1) % LAT_Y;
//         tmp = (fermiField.getField(idx, 0) + flag * i * fermiField.getField(idx, 1)) * Half * gaugeField.getLink(idx, 1, parity);
//         resultField.setField(x + LAT_X * (b_y + LAT_Y * (z + LAT_Z * t)), 0, resultField.getField(x + LAT_X * (b_y + LAT_Y * (z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (b_y + LAT_Y * (z + LAT_Z * t)), 1, resultField.getField(x + LAT_X * (b_y + LAT_Y * (z + LAT_Z * t)), 1) - flag * i * tmp);

//         // forward y
//         int f_y = (y + 1) % LAT_Y;
//         tmp = (fermiField.getField(idx, 0) - flag * i * fermiField.getField(idx, 1)) * Half * conj(gaugeField.getLink(x + LAT_X * (f_y + LAT_Y * (z + LAT_Z * t)), 1, parity));
//         resultField.setField(x + LAT_X * (f_y + LAT_Y * (z + LAT_Z * t)), 0, resultField.getField(x + LAT_X * (f_y + LAT_Y * (z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (f_y + LAT_Y * (z + LAT_Z * t)), 1, resultField.getField(x + LAT_X * (f_y + LAT_Y * (z + LAT_Z * t)), 1) + flag * i * tmp);

//         // backward z
//         int b_z = (z + LAT_Z - 1) % LAT_Z;
//         tmp = (fermiField.getField(idx, 0) + flag * fermiField.getField(idx, 1)) * Half * gaugeField.getLink(idx, 2, parity);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (b_z + LAT_Z * t)), 0, resultField.getField(x + LAT_X * (y + LAT_Y * (b_z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (b_z + LAT_Z * t)), 1, resultField.getField(x + LAT_X * (y + LAT_Y * (b_z + LAT_Z * t)), 1) + flag * tmp);

//         // forward z
//         int f_z = (z + 1) % LAT_Z;
//         tmp = (fermiField.getField(idx, 0) - flag * fermiField.getField(idx, 1)) * Half * conj(gaugeField.getLink(x + LAT_X * (y + LAT_Y * (f_z + LAT_Z * t)), 2, parity));
//         resultField.setField(x + LAT_X * (y + LAT_Y * (f_z + LAT_Z * t)), 0, resultField.getField(x + LAT_X * (y + LAT_Y * (f_z + LAT_Z * t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (f_z + LAT_Z * t)), 1, resultField.getField(x + LAT_X * (y + LAT_Y * (f_z + LAT_Z * t)), 1) - flag * tmp);

//         // backward t
//         int b_t = (t + LAT_T - 1) % LAT_T;
//         tmp = (fermiField.getField(idx, 0) + flag * i * fermiField.getField(idx, 1)) * Half * gaugeField.getLink(idx, 3, parity);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * b_t)), 0, resultField.getField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * b_t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * b_t)), 1, resultField.getField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * b_t)), 1) - flag * i * tmp);

//         // forward t
//         int f_t = (t + 1) % LAT_T;
//         tmp = (fermiField.getField(idx, 0) - flag * i * fermiField.getField(idx, 1)) * Half * conj(gaugeField.getLink(x + LAT_X * (y + LAT_Y * (z + LAT_Z * f_t)), 3, parity));
//         resultField.setField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * f_t)), 0, resultField.getField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * f_t)), 0) + tmp);
//         resultField.setField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * f_t)), 1, resultField.getField(x + LAT_X * (y + LAT_Y * (z + LAT_Z * f_t)), 1) + flag * i * tmp);
//     }
// }
// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "Use GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "Number of SMs:" << devProp.multiProcessorCount << std::endl;
    std::cout << "Shared memory size of each thread block:" << devProp.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
    std::cout << "The maximum number of threads per thread block:" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "The maximum number of threads per EM:" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum number of warps per EM:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);
    // // Allocate memory on the host
    // Complex *hostFermiField = new Complex[NUM_PARITIES * VOLUME];
    // Complex *hostGaugeField = new Complex[4 * NUM_PARITIES * VOLUME];
    // curandState *hostRandomStates = new curandState[VOLUME];

    // // Allocate memory on the device
    // Complex *devFermiField;
    // Complex *devGaugeField;
    // curandState *devRandomStates;
    // cudaMalloc(&devFermiField, sizeof(Complex) * NUM_PARITIES * VOLUME);
    // cudaMalloc(&devGaugeField, sizeof(Complex) * 4 * NUM_PARITIES * VOLUME);
    // cudaMalloc(&devRandomStates, sizeof(curandState) * VOLUME);

    // // Initialize random number generator states on the device
    // int numThreads = 256;
    // int numBlocks = (VOLUME + numThreads - 1) / numThreads;
    // setupRandomGenerator<<<numBlocks, numThreads>>>(devRandomStates);
    // cudaDeviceSynchronize();

    // // Initialize input fields on the device
    // initInputFields<<<numBlocks, numThreads>>>(devFermiField, devGaugeField, devRandomStates);
    // cudaDeviceSynchronize();

    // // Copy input fields from device to host
    // cudaMemcpy(hostFermiField, devFermiField, sizeof(Complex) * NUM_PARITIES * VOLUME, cudaMemcpyDeviceToHost);
    // cudaMemcpy(hostGaugeField, devGaugeField, sizeof(Complex) * 4 * NUM_PARITIES * VOLUME, cudaMemcpyDeviceToHost);

    // // Perform Dslash4 operation on the device
    // FermiField fermiField(devFermiField, NUM_PARITIES);
    // GaugeField gaugeField(devGaugeField, NUM_PARITIES);
    // FermiField resultField(devFermiField, NUM_PARITIES);

    // Dslash4<<<numBlocks, numThreads>>>(fermiField, gaugeField, resultField, 0);
    // cudaDeviceSynchronize();

    // // Copy result field from device to host
    // cudaMemcpy(hostFermiField, devFermiField, sizeof(Complex) * NUM_PARITIES * VOLUME, cudaMemcpyDeviceToHost);

    // // Output results
    // for (int p = 0; p < NUM_PARITIES; p++)
    // {
    //     for (int x = 0; x < LAT_X; x++)
    //     {
    //         for (int y = 0; y < LAT_Y; y++)
    //         {
    //             for (int z = 0; z < LAT_Z; z++)
    //             {
    //                 for (int t = 0; t < LAT_T; t++)
    //                 {
    //                     Complex value = hostFermiField[p * VOLUME + x + LAT_X * (y + LAT_Y * (z + LAT_Z * t))];
    //                     std::cout << "Result[" << p << "][" << x << "][" << y << "][" << z << "][" << t
    //                               << "]: (" << value.real << ", " << value.imag << ")" << std::endl;
    //                 }
    //             }
    //         }
    //     }
    // }

    // // Free memory on the device
    // cudaFree(devFermiField);
    // cudaFree(devGaugeField);
    // cudaFree(devRandomStates);

    // // Free memory on the host
    // delete[] hostFermiField;
    // delete[] hostGaugeField;
    // delete[] hostRandomStates;
}
