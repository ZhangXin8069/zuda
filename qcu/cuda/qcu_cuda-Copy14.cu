#include <filesystem>
#include <limits>
#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <chrono>
#include <cmath>
#include <cstdio>

#define BLOCK_SIZE 256

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
  // just wilson(Sum part)
  {
    for (int i = 0; i < 12; i++) {
      dest[i] = zero;
    }
  }
  {
    // x-1
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
    // x+1
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
    // y-1
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
    // y+1
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
    // z-1
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
    // z+1
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
    // t-1
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
    // t+1
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
__global__ void clover(void *device_U, void *device_clover, int device_lat_x,
                       const int device_lat_y, const int device_lat_z,
                       const int device_lat_t, const int device_parity) {
  register int parity = blockIdx.x * blockDim.x + threadIdx.x;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  register int move0;
  register int move1;
  move0 = lat_x * lat_y * lat_z;
  const int t = parity / move0;
  parity -= t * move0;
  move0 = lat_x * lat_y;
  const int z = parity / move0;
  parity -= z * move0;
  const int y = parity / lat_x;
  const int x = parity - y * lat_x;
  const int oe = (y + z + t) % 2;
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex tmp0(0.0, 0.0);
  register LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  register LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  register LatticeComplex *tmp_U;
  register LatticeComplex tmp1[9];
  register LatticeComplex tmp2[9];
  register LatticeComplex tmp3[9];
  register LatticeComplex U[9];
  register LatticeComplex clover[144];
  // sigmaF
  {
    parity = device_parity;
    for (int i = 0; i < 144; i++) {
      clover[i] = zero;
      origin_clover[i] = zero;
    }
    for (int i = 0; i < 9; i++) {
      tmp1[i] = zero;
      tmp2[i] = zero;
    }
  }
  // XY
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x+1,y,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y+1,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y-1,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y-1,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y-1,z,t;x
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x+1,y-1,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  // XZ
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x+1,y,z,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y,z+1,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y,z-1,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z-1,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z-1,t;x
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x+1,y,z-1,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[126 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
      }
    }
  }
  // XT
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x+1,y,z,t;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y,z,t+1;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x-1,y,z,t-1;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x-1,y,z,t-1;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z,t-1;x
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x+1,y,z,t-1;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YZ
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y+1,z,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y-1,z+1,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y-1,z-1,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z-1,t;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z-1,t;y
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y+1,z-1,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[36 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[99 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
      }
    }
  }
  // YT
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y+1,z,t;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y-1,z,t+1;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y-1,z,t-1;t;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y-1,z,t-1;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z,t-1;y
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y+1,z,t-1;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[9 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
        clover[36 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[99 + c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj());
        clover[126 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-1);
      }
    }
  }

  // ZT
  {
    for (int i = 0; i < 9; i++) {
      U[i] = zero;
    }
  }
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z+1,t;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z-1,t+1;z;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z-1,t-1;t;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 +=
              tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj(); // dag;dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z-1,t-1;z
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    //// x,y,z,t-1;z
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp2[i] = tmp_U[i];
    }
    tmp2[6] = (tmp2[1] * tmp2[5] - tmp2[2] * tmp2[4]).conj();
    tmp2[7] = (tmp2[2] * tmp2[3] - tmp2[0] * tmp2[5]).conj();
    tmp2[8] = (tmp2[0] * tmp2[4] - tmp2[1] * tmp2[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1]; // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z+1,t-1;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp3[c0 * 3 + cc] * tmp1[cc * 3 + c1];
        }
        tmp2[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      tmp1[i] = tmp_U[i];
    }
    tmp1[6] = (tmp1[1] * tmp1[5] - tmp1[2] * tmp1[4]).conj();
    tmp1[7] = (tmp1[2] * tmp1[3] - tmp1[0] * tmp1[5]).conj();
    tmp1[8] = (tmp1[0] * tmp1[4] - tmp1[1] * tmp1[3]).conj();
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 = zero;
        for (int cc = 0; cc < 3; cc++) {
          tmp0 += tmp2[c0 * 3 + cc] * tmp1[c1 * 3 + cc].conj(); // dag
        }
        tmp3[c0 * 3 + c1] = tmp0;
      }
    }
  }
  {
    for (int i = 0; i < 9; i++) {
      U[i] += tmp3[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      for (int c1 = 0; c1 < 3; c1++) {
        clover[c0 * 3 + c1] += (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
        clover[45 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[90 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * (-I);
        clover[135 + c0 * 3 + c1] +=
            (U[c0 * 3 + c1] - U[c1 * 3 + c0].conj()) * I;
      }
    }
  }
  {
    for (int i = 0; i < 144; i++) {
      origin_clover[i] = clover[i] * 0.25;
    }
  }
}
__global__ void give_clover(void *device_propagator, void *device_src,
                            void *device_dest, int device_lat_x,
                            const int device_lat_y, const int device_lat_z,
                            const int device_lat_t, const int device_parity) {
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int parity = device_parity;
  register LatticeComplex zero(0.0, 0.0);
  register LatticeComplex I(0.0, 1.0);
  register LatticeComplex tmp0(0.0, 0.0);
  register int tmp1;
  register int tmp2 = blockIdx.x * blockDim.x + threadIdx.x;
  tmp1 = lat_x * lat_y * lat_z;
  const int t = tmp2 / tmp1;
  tmp2 -= t * tmp1;
  tmp1 = lat_x * lat_y;
  const int z = tmp2 / tmp1;
  tmp2 -= z * tmp1;
  const int y = tmp2 / lat_x;
  const int x = tmp2 - y * lat_x;
  register LatticeComplex *origin_propagator =
      ((static_cast<LatticeComplex *>(device_propagator)) +
       t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
       y * lat_x * 144 + x * 144);
  register LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12 +
       (parity * 2 - 1) * lat_t * lat_z * lat_y * lat_x * 12);
  register LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  register LatticeComplex propagator[144];
  register LatticeComplex src[12];
  register LatticeComplex dest[12];
  register LatticeComplex tmp[12];
  {
    for (int i = 0; i < 144; i++) {
      propagator[i] = origin_propagator[i];
    }
    for (int i = 0; i < 12; i++) {
      src[i] = origin_src[i];
      dest[i] = origin_dest[i];
      tmp[i] = zero;
    }
  }
  {
    for (int sc0 = 0; sc0 < 12; sc0++) {
      tmp0 = zero;
      for (int sc1 = 0; sc1 < 12; sc1++) {
        tmp0 += propagator[sc0 * 12 + sc1] * src[sc1];
      }
      tmp[sc0] = tmp0;
    }
  }
  {
    for (int i = 0; i < 12; i++) {
      origin_dest[i] = src[i] - dest[i] - tmp[i] * 0.25 * I;
    }
  }
}
void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  void *propagator;
  checkCudaErrors(
      cudaMalloc(&propagator, (lat_t * lat_z * lat_y * lat_x * 144) *
                                  sizeof(LatticeComplex)));
  cudaError_t err;
  dim3 gridDim(lat_x * lat_y * lat_z * lat_t / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  {
    // wilson dslash
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    dslash<<<gridDim, blockDim>>>(gauge, fermion_in, fermion_out, lat_x, lat_y,
                                  lat_z, lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.9lf sec\n",
        double(duration) / 1e9);
  }
  {
    // just clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    clover<<<gridDim, blockDim>>>(gauge, propagator, lat_x, lat_y, lat_z, lat_t,
                                  parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("just clover total time: (without malloc free memcpy) : %.9lf sec\n",
           double(duration) / 1e9);
  }
  {
    // give clover
    checkCudaErrors(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    give_clover<<<gridDim, blockDim>>>(propagator, fermion_in, fermion_out,
                                       lat_x, lat_y, lat_z, lat_t, parity);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "give clover total time: (without malloc free memcpy) : %.9lf sec\n
        ", double(duration) / 1e9);
  }
  {
    // free
    checkCudaErrors(cudaFree(propagator));
  }
}