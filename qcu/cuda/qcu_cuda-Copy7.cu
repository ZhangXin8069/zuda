#include <sys/types.h>
#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdio>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define checkCudaErrors(err)                                                   \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "checkCudaErrors() API error = %04d \"%s\" from file <%s>, "     \
              "line %i.\n",                                                    \
              err, cudaGetErrorString(err), __FILE__, __LINE__);               \
      exit(-1);                                                                \
    }                                                                          \
  }
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
  __forceinline__ __device__ LatticeComplex &operator=(const double &other) {
    real = other;
    imag = 0;
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
  __forceinline__ __device__ bool
  operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  __forceinline__ __device__ bool
  operator!=(const LatticeComplex &other) const {
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
  __forceinline__ __device__ double norm2() const {
    return sqrt(real * real + imag * imag);
  }
};
__global__ void dslash(void *device_U, void *device_src, void *device_dest,
                       int device_lat_x, const int device_lat_y,
                       const int device_lat_z, const int device_lat_t,
                       const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  const int lat_xsc = lat_x * 12;
  const int lat_yxsc = lat_y * lat_xsc;
  const int lat_zyxsc = lat_z * lat_yxsc;
  register int move;
  move = lat_x * lat_y * lat_z;
  const int t = parity / move;
  parity -= t * move;
  move = lat_x * lat_y;
  const int z = parity / move;
  parity -= z * move;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  parity = device_parity;
  const int oe = (y + z + t) % 2;
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  register LatticeComplex *tmp_U;
  register LatticeComplex *tmp_src;
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex tmp1(0.0, 0.0);
  register LatticeComplex U[9];
  register LatticeComplex src[12];
  register LatticeComplex dest[12];

  for (int i = 0; i < 12; i++) {
    dest[i] = zero;
  }
  // mass term and others
  // for (int s = 0; s < 4; s++)
  // {
  //     for (int c = 0; c < 3; c++)
  //     {
  //         dest(x, y, z, t, s, c) += src(x, y, z, t, s, c) * 0;
  //     }
  // }

  {
    // backward x
    // if (x == 0) {
    //   move = lat_x - 1;
    // } else {
    //   move = -1;
    // }
    move = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * 12);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp1 * I;
      dest[c0 + 9] -= tmp0 * I;
    }
  }
  {
    // forward x
    // if (x == lat_x - 1) {
    //   move = 1 - lat_x;
    // } else {
    //   move = 1;
    // }
    move = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * 12);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 9] * I) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 6] * I) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp1 * I;
      dest[c0 + 9] += tmp0 * I;
    }
  }
  {
    // backward y
    // if (y == 0) {
    //   move = lat_y - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_xsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 9]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp1;
      dest[c0 + 9] -= tmp0;
    }
  }
  {
    // // forward y
    // if (y == lat_y - 1) {
    //   move = 1 - lat_y;
    // } else {
    //   move = 1;
    // }
    move = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_xsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 9]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 6]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp1;
      dest[c0 + 9] += tmp0;
    }
  }
  {
    // backward z
    // if (z == 0) {
    //   move = lat_z - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_yxsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6] * I) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] - src[c1 + 9] * I) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0 * I;
      dest[c0 + 9] += tmp1 * I;
    }
  }
  {
    // forward z
    // if (z == lat_z - 1) {
    //   move = 1 - lat_z;
    // } else {
    //   move = 1;
    // }
    move = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_yxsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6] * I) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] + src[c1 + 9] * I) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0 * I;
      dest[c0 + 9] -= tmp1 * I;
    }
  }
  {
    // backward t
    // if (t == 0) {
    //   move = lat_t - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_zyxsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] + src[c1 + 6]) * U[c1 * 3 + c0].conj();
        tmp1 += (src[c1 + 3] + src[c1 + 9]) * U[c1 * 3 + c0].conj();
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] += tmp0;
      dest[c0 + 9] += tmp1;
    }
  }
  {
    // forward t
    // if (t == lat_t - 1) {
    //   move = 1 - lat_t;
    // } else {
    //   move = 1;
    // }
    move = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      U[i] = tmp_U[i];
    }
    U[6] = (U[1] * U[5] - U[2] * U[4]).conj();
    U[7] = (U[2] * U[3] - U[0] * U[5]).conj();
    U[8] = (U[0] * U[4] - U[1] * U[3]).conj();
    tmp_src = (origin_src + move * lat_zyxsc);
    for (int i = 0; i < 12; i++) {
      src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (src[c1] - src[c1 + 6]) * U[c0 * 3 + c1];
        tmp1 += (src[c1 + 3] - src[c1 + 9]) * U[c0 * 3 + c1];
      }
      dest[c0] += tmp0;
      dest[c0 + 3] += tmp1;
      dest[c0 + 6] -= tmp0;
      dest[c0 + 9] -= tmp1;
    }
  }
  // {
  //   __shared__ LatticeComplex warp_output_vec[BLOCK_SIZE * 12];
  //   int warp_thread_rank = threadIdx.x & (WARP_SIZE - 1);
  //   int warp_pos = threadIdx.x >> 5;
  //   LatticeComplex *shared_dest =
  //       (static_cast<LatticeComplex *>(device_dest) +
  //        blockIdx.x * BLOCK_SIZE * 12 + warp_pos * WARP_SIZE * 12);
  //   for (int i = 0; i < 12; i++) {
  //     warp_output_vec[threadIdx.x * 12 + i] = dest[i];
  //   }
  //   __syncthreads();
  //   for (int i = warp_thread_rank; i < WARP_SIZE * 12; i += WARP_SIZE) {
  //     shared_dest[i] = warp_output_vec[warp_pos * WARP_SIZE * 12 + i];
  //   }
  //   __syncthreads();
  // }
  {
    for (int i = 0; i < 12; i++) {
      origin_dest[i] = dest[i];
    }
  }
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x, lat_y,
                                lat_z, lat_t, parity);
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("total time: (without malloc free memcpy) : %.9lf sec\n",
         double(duration) / 1e9);
}