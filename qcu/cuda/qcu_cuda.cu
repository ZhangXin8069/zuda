#include <cstdlib>
#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <time.h>
#define checkCudaErrors(err)
{
  if (err != cudaSuccess) {
    fprintf(stderr,
            "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, cudaGetErrorString(err), __FILE__, __LINE__);
    exit(-1);
  }
}
#define getVecAddr(origin, x, y, z, t, Lx, Ly, Lz, Lt)
((origin) + (((t * Lz + z) * Ly + y) * Lx + x) * NS * NC)
#define getGaugeAddr(origin, direction, x, y, z, t, Lx, Ly, Lz, Lt)
    ((origin) + ((((direction * Lt + t) * Lz + z) * Ly + y) * Lx + x) * NC * NC)

        struct LatticeComplex {
  double real;
  double imag;
  __forceinline__ __device__ LatticeComplex(const double &real = 0.0,
                                            const double &imag = 0.0)
      : real(real), imag(imag) {}
  __forceinline__ __device__ LatticeComplex &
  operator=(const LatticeComplex &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex
  operator+(const LatticeComplex &other) const {
    return LatticeComplex(real + other.real, imag + other.imag);
  }
  __forceinline__ __device__ LatticeComplex
  operator-(const LatticeComplex &other) const {
    return LatticeComplex(real - other.real, imag - other.imag);
  }
  __forceinline__ __device__ LatticeComplex
  operator*(const LatticeComplex &other) const {
    return LatticeComplex(real * other.real - imag * other.imag,
                          real * other.imag + imag * other.real);
  }
  __forceinline__ __device__ LatticeComplex
  operator*(const double &other) const {
    return LatticeComplex(real * other, imag * other);
  }
  __forceinline__ __device__ LatticeComplex
  operator/(const LatticeComplex &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeComplex((real * other.real + imag * other.imag) / denom,
                          (imag * other.real - real * other.imag) / denom);
  }
  __forceinline__ __device__ LatticeComplex
  operator/(const double &other) const {
    return LatticeComplex(real / other, imag / other);
  }
  __forceinline__ __device__ LatticeComplex operator-() const {
    return LatticeComplex(-real, -imag);
  }
  __device__ bool operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  __device__ bool operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  __forceinline__ __device__ LatticeComplex &
  operator+=(const LatticeComplex &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator-=(const LatticeComplex &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator*=(const LatticeComplex &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &
  operator/=(const LatticeComplex &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __forceinline__ __device__ LatticeComplex conj() const {
    return LatticeComplex(real, -imag);
  }
};
struct LatticParam {
  int lat_x;
  int lat_y;
  int lat_z;
  int lat_t;
  int lat_s;
  int lat_c;
  int lat_cc;
  int lat_xcc;
  int lat_yxcc;
  int lat_zyxcc;
  int lat_tzyxcc;
  int lat_stzyxcc;
  int lat_sc;
  int lat_xsc;
  int lat_yxsc;
  int lat_zyxsc;
  int lat_tzyxsc;
  int gauge_size;
  int fermi_size;
};

class Gamme {
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
__forceinline__ __device__ int
index_guage(const int &index_x, const int &index_y, const int &index_z,
            const int &index_t, const int &index_s, const int &index_c0,
            const int &index_c1, const LatticParam *lat_param) {
  register const int lat_c = lat_param->lat_c;
  register const int lat_cc = lat_param->lat_cc;
  register const int lat_xcc = lat_param->lat_xcc;
  register const int lat_yxcc = lat_param->lat_yxcc;
  register const int lat_zyxcc = lat_param->lat_zyxcc;
  register const int lat_tzyxcc = lat_param->lat_tzyxcc;
  return index_s * lat_tzyxcc + index_t * lat_zyxcc + index_z * lat_yxcc +
         index_y * lat_xcc + index_x * lat_cc + index_c0 * lat_c + index_c1;
}
__forceinline__ __device__ int
index_fermi(const int &index_x, const int &index_y, const int &index_z,
            const int &index_t, const int &index_s, const int &index_c,
            const LatticParam *lat_param) {
  register const int lat_c = lat_param->lat_c;
  register const int lat_sc = lat_param->lat_sc;
  register const int lat_xsc = lat_param->lat_xsc;
  register const int lat_yxsc = lat_param->lat_yxsc;
  register const int lat_zyxsc = lat_param->lat_zyxsc;
  return index_t * lat_zyxsc + index_z * lat_yxsc + index_y * lat_xsc +
         index_x * lat_sc + index_s * lat_c + index_c;
}
__global__ void dslash(const LatticeComplex *U, const LatticeComplex *src,
                       LatticeComplex *dest, LatticParam *lat_param) {
  register const int x = blockIdx.x;
  register const int y = blockIdx.y;
  register const int z = blockIdx.z;
  register const int t = threadIdx.x;
  register const int lat_x = lat_param->lat_x;
  register const int lat_y = lat_param->lat_y;
  register const int lat_z = lat_param->lat_z;
  register const int lat_t = lat_param->lat_t;
  register const int lat_s = lat_param->lat_s;
  register const int lat_c = lat_param->lat_c;
  register const LatticeComplex i(0.0, 1.0);
  register const LatticeComplex zero(0.0, 0.0);
  register int tmp;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex local_dest[12];
  for (int c0 = 0; c0 < lat_c; c0++) {
    local_dest[c0 * lat_s + 0] =
        dest[index_fermi(x, y, z, t, 0, c0, lat_param)];
    local_dest[c0 * lat_s + 1] =
        dest[index_fermi(x, y, z, t, 1, c0, lat_param)];
    local_dest[c0 * lat_s + 2] =
        dest[index_fermi(x, y, z, t, 2, c0, lat_param)];
    local_dest[c0 * lat_s + 3] =
        dest[index_fermi(x, y, z, t, 3, c0, lat_param)];
  }

  // mass term and others
  // for (int s = 0; s < lat_s; s++)
  // {
  //     for (int c = 0; c < lat_c; c++)
  //     {
  //         dest(x, y, z, t, s, c) += src(x, y, z, t, s, c) * 0;
  //     }
  // }
  // backward x
  if (x == 0) {
    tmp = lat_x - 1;
  } else {
    tmp = x - 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1, lat_param)] +
               src[index_fermi(tmp, y, z, t, 3, c1, lat_param)] * i) *
              U[index_guage(tmp, y, z, t, 0, c1, c0, lat_param)].conj();
      tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1, lat_param)] +
               src[index_fermi(tmp, y, z, t, 2, c1, lat_param)] * i) *
              U[index_guage(tmp, y, z, t, 0, c1, c0, lat_param)].conj();
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] -= tmp1 * i;
    local_dest[c0 * lat_s + 3] -= tmp0 * i;
  }
  // forward x
  if (x == lat_x - 1) {
    tmp = 0;
  } else {
    tmp = x + 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1, lat_param)] -
               src[index_fermi(tmp, y, z, t, 3, c1, lat_param)] * i) *
              U[index_guage(x, y, z, t, 0, c0, c1, lat_param)];
      tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1, lat_param)] -
               src[index_fermi(tmp, y, z, t, 2, c1, lat_param)] * i) *
              U[index_guage(x, y, z, t, 0, c0, c1, lat_param)];
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] += tmp1 * i;
    local_dest[c0 * lat_s + 3] += tmp0 * i;
  }
  // backward y
  if (y == 0) {
    tmp = lat_y - 1;
  } else {
    tmp = y - 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1, lat_param)] -
               src[index_fermi(x, tmp, z, t, 3, c1, lat_param)]) *
              U[index_guage(x, tmp, z, t, 1, c1, c0, lat_param)].conj();
      tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1, lat_param)] +
               src[index_fermi(x, tmp, z, t, 2, c1, lat_param)]) *
              U[index_guage(x, tmp, z, t, 1, c1, c0, lat_param)].conj();
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] += tmp1;
    local_dest[c0 * lat_s + 3] -= tmp0;
  }
  // forward y
  if (y == lat_y - 1) {
    tmp = 0;
  } else {
    tmp = y + 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1, lat_param)] +
               src[index_fermi(x, tmp, z, t, 3, c1, lat_param)]) *
              U[index_guage(x, y, z, t, 1, c0, c1, lat_param)];
      tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1, lat_param)] -
               src[index_fermi(x, tmp, z, t, 2, c1, lat_param)]) *
              U[index_guage(x, y, z, t, 1, c0, c1, lat_param)];
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] -= tmp1;
    local_dest[c0 * lat_s + 3] += tmp0;
  }
  // backward z
  if (z == 0) {
    tmp = lat_z - 1;
  } else {
    tmp = z - 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1, lat_param)] +
               src[index_fermi(x, y, tmp, t, 2, c1, lat_param)] * i) *
              U[index_guage(x, y, tmp, t, 2, c1, c0, lat_param)].conj();
      tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1, lat_param)] -
               src[index_fermi(x, y, tmp, t, 3, c1, lat_param)] * i) *
              U[index_guage(x, y, tmp, t, 2, c1, c0, lat_param)].conj();
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] -= tmp0 * i;
    local_dest[c0 * lat_s + 3] += tmp1 * i;
  }
  // forward z
  if (z == lat_z - 1) {
    tmp = 0;
  } else {
    tmp = z + 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1, lat_param)] -
               src[index_fermi(x, y, tmp, t, 2, c1, lat_param)] * i) *
              U[index_guage(x, y, z, t, 2, c0, c1, lat_param)];
      tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1, lat_param)] +
               src[index_fermi(x, y, tmp, t, 3, c1, lat_param)] * i) *
              U[index_guage(x, y, z, t, 2, c0, c1, lat_param)];
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] += tmp0 * i;
    local_dest[c0 * lat_s + 3] -= tmp1 * i;
  }
  // backward t
  if (t == 0) {
    tmp = lat_t - 1;
  } else {
    tmp = t - 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1, lat_param)] +
               src[index_fermi(x, y, z, tmp, 2, c1, lat_param)]) *
              U[index_guage(x, y, z, tmp, 3, c1, c0, lat_param)].conj();
      tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1, lat_param)] +
               src[index_fermi(x, y, z, tmp, 3, c1, lat_param)]) *
              U[index_guage(x, y, z, tmp, 3, c1, c0, lat_param)].conj();
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] += tmp0;
    local_dest[c0 * lat_s + 3] += tmp1;
  }
  // forward t
  if (t == lat_t - 1) {
    tmp = 0;
  } else {
    tmp = t + 1;
  }
  for (int c0 = 0; c0 < lat_c; c0++) {
    tmp0 = zero;
    tmp1 = zero;
    for (int c1 = 0; c1 < lat_c; c1++) {
      tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1, lat_param)] -
               src[index_fermi(x, y, z, tmp, 2, c1, lat_param)]) *
              U[index_guage(x, y, z, t, 3, c0, c1, lat_param)];
      tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1, lat_param)] -
               src[index_fermi(x, y, z, tmp, 3, c1, lat_param)]) *
              U[index_guage(x, y, z, t, 3, c0, c1, lat_param)];
    }
    local_dest[c0 * lat_s + 0] += tmp0;
    local_dest[c0 * lat_s + 1] += tmp1;
    local_dest[c0 * lat_s + 2] -= tmp0;
    local_dest[c0 * lat_s + 3] -= tmp1;
  }

  for (int c0 = 0; c0 < lat_c; c0++) {
    dest[index_fermi(x, y, z, t, 0, c0, lat_param)] =
        local_dest[c0 * lat_s + 0];
    dest[index_fermi(x, y, z, t, 1, c0, lat_param)] =
        local_dest[c0 * lat_s + 1];
    dest[index_fermi(x, y, z, t, 2, c0, lat_param)] =
        local_dest[c0 * lat_s + 2];
    dest[index_fermi(x, y, z, t, 3, c0, lat_param)] =
        local_dest[c0 * lat_s + 3];
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
// __host__ LatticeComplex dot(const LatticeComplex *a, const LatticeComplex *b,
// const int &size)
// {
//     LatticeComplex result(0.0, 0.0);
//     for (int i = 0; i < size; i++)
//     {
//         result += a[i].conj() * b[i];
//     }
//     return result;
// }

__host__ void assign_zero(LatticeComplex *a, const int &size) {
  for (int i = 0; i < size; i++) {
    a[i].real = 0;
    a[i].imag = 0;
  }
}

__host__ void assign_unit(LatticeComplex *a, const int &size) {
  for (int i = 0; i < size; i++) {
    a[i].real = 1;
    a[i].imag = 0;
  }
}

__host__ void assign_random(LatticeComplex *a, const int &size,
                            const unsigned &seed) {
  std::default_random_engine e(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int i = 0; i < size; i++) {
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

__host__ void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
                        QcuParam *param) {

  LatticParam *lat_param;
  LatticeComplex *U, *src, *dest;
  lat_param = (LatticParam *)malloc(sizeof(LatticParam));
  U = (LatticeComplex *)malloc(gauge_size * sizeof(LatticeComplex));
  src = (LatticeComplex *)malloc(fermi_size * sizeof(LatticeComplex));
  dest = (LatticeComplex *)malloc(fermi_size * sizeof(LatticeComplex));
  lat_param->lat_x = param->lattice_size[0];
  lat_param->lat_y = param->lattice_size[1];
  lat_param->lat_z = param->lattice_size[2];
  lat_param->lat_t = param->lattice_size[3];
  lat_param->lat_s = 4;
  lat_param->lat_c = 3;
  lat_param->lat_cc = lat_param->lat_c * lat_param->lat_c;
  lat_param->lat_xcc = lat_param->lat_x * lat_param->lat_cc;
  lat_param->lat_yxcc = lat_param->lat_y * lat_param->lat_xcc;
  lat_param->lat_zyxcc = lat_param->lat_z * lat_param->lat_yxcc;
  lat_param->lat_tzyxcc = lat_param->lat_t * lat_param->lat_zyxcc;
  lat_param->lat_stzyxcc = lat_param->lat_s * lat_param->lat_tzyxcc;
  lat_param->lat_sc = lat_param->lat_s * lat_param->lat_c;
  lat_param->lat_xsc = lat_param->lat_x * lat_param->lat_sc;
  lat_param->lat_yxsc = lat_param->lat_y * lat_param->lat_xsc;
  lat_param->lat_zyxsc = lat_param->lat_z * lat_param->lat_yxsc;
  lat_param->lat_tzyxsc = lat_param->lat_t * lat_param->lat_zyxsc;
  lat_param->gauge_size = lat_param->lat_stzyxcc;
  lat_param->fermi_size = lat_param->lat_tzyxsc;
  std::cout << "################"
            << "lat_param->gauge_size\n"
            << lat_param->gauge_size << "lat_param->fermi_size\n"
            << lat_param->fermi_size << std::endl;
  LatticParam *device_lat_param;
  LatticeComplex *device_U, *device_src, *device_dest;
  checkCudaErrors(cudaMalloc((void **)&device_U,
                             lat_param->gauge_size * sizeof(LatticeComplex)));
  checkCudaErrors(cudaMalloc((void **)&device_src,
                             lat_param->fermi_size * sizeof(LatticeComplex)));
  checkCudaErrors(cudaMalloc((void **)&device_dest,
                             lat_param->fermi_size * sizeof(LatticeComplex)));
  checkCudaErrors(cudaMemcpy((void *)device_U, (void *)gauge,
                             lat_param->gauge_size * sizeof(LatticeComplex),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy((void *)device_src, (void *)fermion_in,
                             lat_param->fermi_size * sizeof(LatticeComplex),
                             cudaMemcpyHostToDevice));
  dim3 gridSize(lat_param->lat_x, lat_param->lat_y, lat_param->lat_z);
  dim3 blockSize(lat_param->lat_t);
  clock_t start = clock();
  dslash<<<gridSize, blockSize>>>(device_U, device_src, device_dest, lat_param);
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  clock_t end0 = clock();
  checkCudaErrors(cudaMemcpy((void *)fermion_out, (void *)device_dest,
                             lat_param->fermi_size * sizeof(LatticeComplex),
                             cudaMemcpyDeviceToHost));
  clock_t end1 = clock();
  std::cout << "################"
            << "time cost without cudaMemcpy:"
            << (double)(end0 - start) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << "################"
            << "time cost with cudaMemcpy:"
            << (double)(end1 - start) / CLOCKS_PER_SEC << "s" << std::endl;
  checkCudaErrors(cudaFree(device_U));
  checkCudaErrors(cudaFree(device_src));
  checkCudaErrors(cudaFree(device_dest));
}
