#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <random>
const int lat_x = 32;
const int lat_y = 32;
const int lat_z = 32;
const int lat_t = 32;
const int lat_s = 4;
const int lat_c = 3;
const int lat_c0 = 3;
const int lat_c1 = 3;
const int size_fermi = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c;
const int size_guage = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1;

struct LatticeComplex
{
    // double real;
    // double imag;
    __device__ double data[2];
    __device__ double &real = data[0];
    __device__ double &imag = data[1];
    __device__ LatticeComplex(const double &real, const double &imag) : real(real), imag(imag) {}
    __device__ __forceinline__ LatticeComplex &operator=(const LatticeComplex &other)
    {
        return {other.real, other.imag};
    }
    __device__ __forceinline__ LatticeComplex operator+(const LatticeComplex &other) const
    {
        return {real + other.real, imag + other.imag};
    }
    __device__ __forceinline__ LatticeComplex operator-(const LatticeComplex &other) const
    {
        return {real - other.real, imag - other.imag};
    }
    __device__ __forceinline__ LatticeComplex operator*(const LatticeComplex &other) const
    {
        return {real * other.real - imag * other.imag,
                real * other.imag + imag * other.real};
    }
    __device__ __forceinline__ LatticeComplex operator*(const double &other) const
    {
        return {real * other, imag * other};
    }
    __device__ __forceinline__ LatticeComplex operator/(const LatticeComplex &other) const
    {
        double denom = other.real * other.real + other.imag * other.imag;
        return {(real * other.real + imag * other.imag) / denom,
                (imag * other.real - real * other.imag) / denom};
    }
    __device__ __forceinline__ LatticeComplex operator/(const double &other) const
    {
        return {real / other, imag / other};
    }
    __device__ __forceinline__ LatticeComplex operator-() const
    {
        return {-real, -imag};
    }
    __device__ bool operator==(const LatticeComplex &other) const
    {
        return (real == other.real && imag == other.imag);
    }
    __device__ bool operator!=(const LatticeComplex &other) const
    {
        return !(*this == other);
    }
    __device__ __forceinline__ LatticeComplex &operator+=(const LatticeComplex &other) const
    {
        return {real + other.real, imag + other.imag};
    }
    __device__ __forceinline__ LatticeComplex &operator-=(const LatticeComplex &other) const
    {
        return {real - other.real, imag - other.imag};
    }
    __device__ __forceinline__ LatticeComplex &operator*=(const LatticeComplex &other) const
    {
        return {real * other.real - imag * other.imag,
                real * other.imag + imag * other.real};
    }
    __device__ __forceinline__ LatticeComplex &operator*=(const double &other) const
    {
        return {real * other, imag * other};
    }
    __device__ __forceinline__ LatticeComplex &operator/=(const LatticeComplex &other) const
    {
        double denom = other.real * other.real + other.imag * other.imag;
        return {(real * other.real + imag * other.imag) / denom,
                (imag * other.real - real * other.imag) / denom};
    }
    __device__ __forceinline__ LatticeComplex &operator/=(const double &other) const
    {
        return {real / other, imag / other};
    }
    __device__ __forceinline__ LatticeComplex conj()
    {
        return {real, -imag};
    }
};
const LatticeComplex i(0.0, 1.0);
const LatticeComplex zero(0.0, 0.0);

__device__ __forceinline__ int index_guage(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
{
    __device__ int tmp(1), result(0);
    result += index_c1;
    tmp *= lat_c1;
    result += index_c0 * tmp;
    tmp *= lat_c0;
    result += index_s * tmp;
    tmp *= lat_s;
    result += index_t * tmp;
    tmp *= lat_t;
    result += index_z * tmp;
    tmp *= lat_z;
    result += index_y * tmp;
    tmp *= lat_y;
    result += index_x * tmp;
    return result;

    return index_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1 + index_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1 + index_z * lat_t * lat_s * lat_c0 * lat_c1 + index_t * lat_s * lat_c0 * lat_c1 + index_s * lat_c0 * lat_c1 + index_c0 * lat_c1 + index_c1;
}
__device__ __forceinline__ int index_fermi(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
{
    return index_x * lat_y * lat_z * lat_t * lat_s * lat_c + index_y * lat_z * lat_t * lat_s * lat_c + index_z * lat_t * lat_s * lat_c + index_t * lat_s * lat_c + index_s * lat_c + index_c;
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
__global__ void dslash(LatticeComplex *U, LatticeFermi *src, LatticeFermi *dest)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int t = threadIdx.x;
    int tmp;
    LatticeComplex tmp0;
    LatticeComplex tmp1;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0 * i;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0 * i;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1 * i;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0 * i;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1 * i;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1;
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
        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0;
        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1;
    }
}
double norm_2()
{
    double result = 0;
    for (int i = 0; i < this->size; i++)
    {
        result = result + this->lattice_vec[i].data[0] * this->lattice_vec[i].data[0] + this->lattice_vec[i].data[1] * this->lattice_vec[i].data[1];
    }
    return result;
}
Complex dot(const LatticeGauge &other)
{
    Complex result;
    for (int i = 0; i < this->size; i++)
    {
        result = result + this->lattice_vec[i].conj() * other[i];
    }
    return result;
}
int main()
{
    const int N = 1024;
    LatticeComplex *a, *b;

    cudaMalloc(&a, N * sizeof(LatticeComplex));
    cudaMalloc(&b, N * sizeof(LatticeComplex));

    // 初始化 a, b

    kernel<<<1, N>>>(a, b);

    cudaFree(a);
    cudaFree(b);

    int size_guage = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1;
    int size_fermi = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c;
    LatticeGauge *_U;
    LatticeFermi *_src;
    LatticeFermi *_dest;
    cudaMallocManaged(&_U, sizeof(LatticeGauge));
    cudaMallocManaged(&_src, sizeof(LatticeFermi));
    cudaMallocManaged(&_dest, sizeof(LatticeFermi));
    cudaMallocManaged(&_U->lattice_vec, lat_c * size * 2 * sizeof(double));
    cudaMallocManaged(&_src->lattice_vec, size * 2 * sizeof(double));
    cudaMallocManaged(&_dest->lattice_vec, size * 2 * sizeof(double));
    LatticeComplex *U = *_U;
    LatticeFermi &src = *_src;
    LatticeFermi &dest = *_dest;
    U.assign_random(666);
    src.assign_random(111);
    dest.assign_zero();
    dim3 gridSize(lat_x, lat_y, lat_z);
    dim3 blockSize(lat_t);
    std::cout << "src.norm_2():" << src.norm_2() << std::endl;
    std::cout << "dest.norm_2():" << dest.norm_2() << std::endl;
    clock_t start = clock();
    dslash<<<gridSize, blockSize>>>(U, src, dest);
    cudaDeviceSynchronize();
    clock_t end = clock();
    std::cout << "src.norm_2():" << src.norm_2() << std::endl;
    std::cout << "dest.norm_2():" << dest.norm_2() << std::endl;
    std::cout
        << "################"
        << "time cost:"
        << (double)(end - start) / CLOCKS_PER_SEC
        << "s"
        << std::endl;
    //// MPI_Finalize();
    return 0;
}