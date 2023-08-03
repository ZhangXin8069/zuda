#pragma nv_verbose
#pragma optimize(5)
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <random>
__device__ const int lat_x = 32;
__device__ const int lat_y = 32;
__device__ const int lat_z = 32;
__device__ const int lat_t = 64;
__device__ const int lat_s = 4;
__device__ const int lat_c = 3;
__device__ const int lat_c0 = 3;
__device__ const int lat_c1 = 3;
__device__ const int size_guage = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1;
__device__ const int size_fermi = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c;
__device__ const int tmp_c = lat_c;
__device__ const int tmp_sc = lat_s * tmp_c;
__device__ const int tmp_tsc = lat_t * tmp_sc;
__device__ const int tmp_ztsc = lat_z * tmp_tsc;
__device__ const int tmp_yztsc = lat_y * tmp_ztsc;
__device__ const int tmp_cc = lat_c0 * lat_c1;
__device__ const int tmp_scc = lat_s * tmp_cc;
__device__ const int tmp_tscc = lat_t * tmp_scc;
__device__ const int tmp_ztscc = lat_z * tmp_tscc;
__device__ const int tmp_yztscc = lat_y * tmp_ztscc;
struct LatticeComplex
{
    double real;
    double imag;
    __device__ __forceinline__ LatticeComplex(const double &real = 0.0, const double &imag = 0.0) : real(real), imag(imag) {}
    __device__ __forceinline__ LatticeComplex &operator=(const LatticeComplex &other)
    {
        real = other.real;
        imag = other.imag;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex operator+(const LatticeComplex &other) const
    {
        return LatticeComplex(real + other.real, imag + other.imag);
    }
    __device__ __forceinline__ LatticeComplex operator-(const LatticeComplex &other) const
    {
        return LatticeComplex(real - other.real, imag - other.imag);
    }
    __device__ __forceinline__ LatticeComplex operator*(const LatticeComplex &other) const
    {
        return LatticeComplex(real * other.real - imag * other.imag,
                              real * other.imag + imag * other.real);
    }
    __device__ __forceinline__ LatticeComplex operator*(const double &other) const
    {
        return LatticeComplex(real * other, imag * other);
    }
    __device__ __forceinline__ LatticeComplex operator/(const LatticeComplex &other) const
    {
        double denom = other.real * other.real + other.imag * other.imag;
        return LatticeComplex((real * other.real + imag * other.imag) / denom,
                              (imag * other.real - real * other.imag) / denom);
    }
    __device__ __forceinline__ LatticeComplex operator/(const double &other) const
    {
        return LatticeComplex(real / other, imag / other);
    }
    __device__ __forceinline__ LatticeComplex operator-() const
    {
        return LatticeComplex(-real, -imag);
    }
    __device__ bool operator==(const LatticeComplex &other) const
    {
        return (real == other.real && imag == other.imag);
    }
    __device__ bool operator!=(const LatticeComplex &other) const
    {
        return !(*this == other);
    }
    __device__ __forceinline__ LatticeComplex &operator+=(const LatticeComplex &other)
    {
        real = real + other.real;
        imag = imag + other.imag;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex &operator-=(const LatticeComplex &other)
    {
        real = real - other.real;
        imag = imag - other.imag;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex &operator*=(const LatticeComplex &other)
    {
        real = real * other.real - imag * other.imag;
        imag = real * other.imag + imag * other.real;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex &operator*=(const double &other)
    {
        real = real * other;
        imag = imag * other;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex &operator/=(const LatticeComplex &other)
    {
        double denom = other.real * other.real + other.imag * other.imag;
        real = (real * other.real + imag * other.imag) / denom;
        imag = (imag * other.real - real * other.imag) / denom;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex &operator/=(const double &other)
    {
        real = real / other;
        imag = imag / other;
        return *this;
    }
    __device__ __forceinline__ LatticeComplex conj() const
    {
        return LatticeComplex(real, -imag);
    }
};

__device__ __forceinline__ int index_guage(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
{
    return index_x * tmp_yztscc + index_y * tmp_ztscc + index_z * tmp_tscc + index_t * tmp_scc + index_s * tmp_cc + index_c0 * tmp_c + index_c1;
}

__device__ __forceinline__ int index_fermi(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
{
    return index_x * tmp_yztsc + index_y * tmp_ztsc + index_z * tmp_tsc + index_t * tmp_sc + index_s * tmp_c + index_c;
}

class Gamme
{
    /*
    Gamme0=
    [[0,0,0,i],
    [0,0,i,0],
    [0,-i,0,0],
    [-i,0,0,0]]
    Gamme1=
    [[0,0,0,-1],
    [0,0,1,0],
    [0,1,0,0],
    [-1,0,0,0]]
    Gamme2=
    [[0,0,i,0],
    [0,0,0,-i],
    [-i,0,0,0],
    [0,i,0,0]]
    Gamme3=
    [[0,0,1,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0]]
    */
};

__global__ void dslash(const LatticeComplex *U, const LatticeComplex *src, LatticeComplex *dest)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int t = threadIdx.x;
    const LatticeComplex i(0.0, 1.0);
    const LatticeComplex zero(0.0, 0.0);
    int tmp;
    LatticeComplex tmp0(0.0, 0.0);
    LatticeComplex tmp1(0.0, 0.0);
    LatticeComplex local_dest[12];
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        local_dest[c0 * lat_s + 0] = dest[index_fermi(x, y, z, t, 0, c0)];
        local_dest[c0 * lat_s + 1] = dest[index_fermi(x, y, z, t, 1, c0)];
        local_dest[c0 * lat_s + 2] = dest[index_fermi(x, y, z, t, 2, c0)];
        local_dest[c0 * lat_s + 3] = dest[index_fermi(x, y, z, t, 3, c0)];
    }

    // mass term and others
    // for (int s = 0; s < lat_s; s++)
    // {
    //     for (int c = 0; c < lat_c0; c++)
    //     {
    //         dest(x, y, z, t, s, c) += src(x, y, z, t, s, c) * 0;
    //     }
    // }
    // backward x
    if (x == 0)
    {
        tmp = lat_x - 1;
    }
    else
    {
        tmp = x - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] + src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] + src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] -= tmp1 * i;
        local_dest[c0 * lat_s + 3] -= tmp0 * i;
    }
    // forward x
    if (x == lat_x - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = x + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] - src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] - src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] += tmp1 * i;
        local_dest[c0 * lat_s + 3] += tmp0 * i;
    }
    // backward y
    if (y == 0)
    {
        tmp = lat_y - 1;
    }
    else
    {
        tmp = y - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] - src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] + src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] += tmp1;
        local_dest[c0 * lat_s + 3] -= tmp0;
    }
    // forward y
    if (y == lat_y - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = y + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] + src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] - src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] -= tmp1;
        local_dest[c0 * lat_s + 3] += tmp0;
    }
    // backward z
    if (z == 0)
    {
        tmp = lat_z - 1;
    }
    else
    {
        tmp = z - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] + src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] - src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] -= tmp0 * i;
        local_dest[c0 * lat_s + 3] += tmp1 * i;
    }
    // forward z
    if (z == lat_z - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = z + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] - src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] + src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] += tmp0 * i;
        local_dest[c0 * lat_s + 3] -= tmp1 * i;
    }
    // backward t
    if (t == 0)
    {
        tmp = lat_t - 1;
    }
    else
    {
        tmp = t - 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] + src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] + src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] += tmp0;
        local_dest[c0 * lat_s + 3] += tmp1;
    }
    // forward t
    if (t == lat_t - 1)
    {
        tmp = 0;
    }
    else
    {
        tmp = t + 1;
    }
    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        tmp0 = zero;
        tmp1 = zero;
        for (int c1 = 0; c1 < lat_c1; c1++)
        {
            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] - src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] - src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
        }
        local_dest[c0 * lat_s + 0] += tmp0;
        local_dest[c0 * lat_s + 1] += tmp1;
        local_dest[c0 * lat_s + 2] -= tmp0;
        local_dest[c0 * lat_s + 3] -= tmp1;
    }

    for (int c0 = 0; c0 < lat_c0; c0++)
    {
        dest[index_fermi(x, y, z, t, 0, c0)] = local_dest[c0 * lat_s + 0];
        dest[index_fermi(x, y, z, t, 1, c0)] = local_dest[c0 * lat_s + 1];
        dest[index_fermi(x, y, z, t, 2, c0)] = local_dest[c0 * lat_s + 2];
        dest[index_fermi(x, y, z, t, 3, c0)] = local_dest[c0 * lat_s + 3];
    }
}

// __host__ double norm_2(const LatticeComplex *a, const int &size)
// {
//     double result = 0;
//     for (int i = 0; i < size; i++)
//     {
//         result += a[i].real * a[i].real + a[i].imag * a[i].imag;
//     }
//     return result;
// }
// __host__ LatticeComplex dot(const LatticeComplex *a, const LatticeComplex *b, const int &size)
// {
//     LatticeComplex result(0.0, 0.0);
//     for (int i = 0; i < size; i++)
//     {
//         result += a[i].conj() * b[i];
//     }
//     return result;
// }

__host__ void assign_zero(LatticeComplex *a, const int &size)
{
    for (int i = 0; i < size; i++)
    {
        a[i].real = 0;
        a[i].imag = 0;
    }
}

__host__ void assign_unit(LatticeComplex *a, const int &size)
{
    for (int i = 0; i < size; i++)
    {
        a[i].real = 1;
        a[i].imag = 0;
    }
}

__host__ void assign_random(LatticeComplex *a, const int &size, const unsigned &seed)
{
    std::default_random_engine e(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < size; i++)
    {
        a[i].real = u(e);
        a[i].imag = u(e);
    }
}

// __global__ void assign_random(LatticeComplex *a, int size, unsigned seed)
// {

//     curandState state;
//     curandevice_init(seed, index, 0, &state);
//     a[index].real = curand_uniform(&state);
//     a[index].imag = curand_uniform(&state);
// }

int main()
{

    LatticeComplex *U, *src, *dest;
    U = (LatticeComplex *)malloc(size_guage * sizeof(LatticeComplex));
    src = (LatticeComplex *)malloc(size_fermi * sizeof(LatticeComplex));
    dest = (LatticeComplex *)malloc(size_fermi * sizeof(LatticeComplex));
    assign_random(U, size_guage, 66);
    assign_random(src, size_guage, 77);
    assign_zero(dest, size_guage);
    std::cout << "sizeof(LatticeComplex)" << sizeof(LatticeComplex) << std::endl;
    std::cout << "sizeof(double)" << sizeof(double) << std::endl;
    LatticeComplex *device_U, *device_src, *device_dest;
    cudaMalloc((void **)&device_U, size_guage * sizeof(LatticeComplex));
    cudaMalloc((void **)&device_src, size_fermi * sizeof(LatticeComplex));
    cudaMalloc((void **)&device_dest, size_fermi * sizeof(LatticeComplex));
    cudaMemcpy((void *)device_U, (void *)U, size_guage * sizeof(LatticeComplex), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)device_src, (void *)src, size_fermi * sizeof(LatticeComplex), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)device_dest, (void *)dest, size_fermi * sizeof(LatticeComplex), cudaMemcpyHostToDevice);
    dim3 gridSize(lat_x, lat_y, lat_z);
    dim3 blockSize(lat_t);
    clock_t start = clock();
    dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest);
    cudaDeviceSynchronize();
    dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest);
    cudaDeviceSynchronize();
    dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest);
    cudaDeviceSynchronize();
    dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest);
    cudaDeviceSynchronize();
    dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest);
    cudaDeviceSynchronize();
    clock_t end0 = clock();
    cudaMemcpy((void *)U, (void *)device_U, size_guage * sizeof(LatticeComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)src, (void *)device_src, size_fermi * sizeof(LatticeComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)dest, (void *)device_dest, size_fermi * sizeof(LatticeComplex), cudaMemcpyDeviceToHost);
    clock_t end1 = clock();
    std::cout
        << "################"
        << "time cost without cudaMemcpy:"
        << (double)(end0 - start) / 5.0 / CLOCKS_PER_SEC
        << "s"
        << std::endl;
    std::cout
        << "################"
        << "time cost with cudaMemcpy:"
        << (double)(end1 - start) / CLOCKS_PER_SEC
        << "s"
        << std::endl;
    cudaFree(device_U);
    cudaFree(device_src);
    cudaFree(device_dest);
    return 0;
}