#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <time.h>

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
  __forceinline__ __device__ double norm2() const {
    return sqrt(real * real + imag * imag);
  }
};
__global__ void dslash(void *device_U, void *device_src, void *device_dest,
                       const int lat_x, const int lat_y, const int lat_z,
                       const int lat_t, const int lat_xcc, const int lat_yxcc,
                       const int lat_zyxcc, const int lat_tzyxcc,
                       const int lat_xsc, const int lat_yxsc,
                       const int lat_zyxsc) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
  const int t = threadIdx.x;
  const int local_lat_xsc = lat_xsc;
  const int local_lat_yxsc = lat_yxsc;
  const int local_lat_zyxsc = lat_zyxsc;
  LatticeComplex *dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * local_lat_zyxsc +
       z * local_lat_yxsc + y * local_lat_xsc + x * 12);
  LatticeComplex local_dest[12];
  const int local_lat_x = lat_x;
  const int local_lat_y = lat_y;
  const int local_lat_z = lat_z;
  const int local_lat_t = lat_t;
  const int local_lat_xcc = lat_xcc;
  const int local_lat_yxcc = lat_yxcc;
  const int local_lat_zyxcc = lat_zyxcc;
  const int local_lat_tzyxcc = lat_tzyxcc;

  const LatticeComplex I(0.0, 1.0);
  const LatticeComplex zero(0.0, 0.0);
  int move;
  double norm;
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * local_lat_zyxcc +
       z * local_lat_yxcc + y * local_lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * local_lat_zyxsc +
       z * local_lat_yxsc + y * local_lat_xsc + x * 12);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex local_U[9];
  LatticeComplex local_src[12];
  for (int i = 0; i < 12; i++) {
    local_dest[i] = zero;
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
    if (x == 0) {
      move = local_lat_x - 1;
    } else {
      move = -1;
    }
    tmp_U = (origin_U + move * 9);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * 12);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] + local_src[c1 + 9] * I) *
                local_U[c1 * 3 + c0].conj();
        tmp1 += (local_src[c1 + 3] + local_src[c1 + 6] * I) *
                local_U[c1 * 3 + c0].conj();
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] -= tmp1 * I;
      local_dest[c0 + 9] -= tmp0 * I;
    }
  }
  {
    // forward x
    if (x == local_lat_x - 1) {
      move = 1 - local_lat_x;
    } else {
      move = 1;
    }
    tmp_U = origin_U;
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * 12);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] - local_src[c1 + 9] * I) * local_U[c0 * 3 + c1];
        tmp1 +=
            (local_src[c1 + 3] - local_src[c1 + 6] * I) * local_U[c0 * 3 + c1];
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] += tmp1 * I;
      local_dest[c0 + 9] += tmp0 * I;
    }
  }
  {
    // backward y
    if (y == 0) {
      move = local_lat_y - 1;
    } else {
      move = -1;
    }
    tmp_U = (origin_U + move * local_lat_xcc + local_lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_xsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 +=
            (local_src[c1] - local_src[c1 + 9]) * local_U[c1 * 3 + c0].conj();
        tmp1 += (local_src[c1 + 3] + local_src[c1 + 6]) *
                local_U[c1 * 3 + c0].conj();
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] += tmp1;
      local_dest[c0 + 9] -= tmp0;
    }
  }
  {
    // forward y
    if (y == local_lat_y - 1) {
      move = 1 - local_lat_y;
    } else {
      move = 1;
    }
    tmp_U = (origin_U + local_lat_tzyxcc);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_xsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] + local_src[c1 + 9]) * local_U[c0 * 3 + c1];
        tmp1 += (local_src[c1 + 3] - local_src[c1 + 6]) * local_U[c0 * 3 + c1];
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] -= tmp1;
      local_dest[c0 + 9] += tmp0;
    }
  }
  {
    // backward z
    if (z == 0) {
      move = local_lat_z - 1;
    } else {
      move = -1;
    }
    tmp_U = (origin_U + move * local_lat_yxcc + local_lat_tzyxcc * 2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_yxsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] + local_src[c1 + 6] * I) *
                local_U[c1 * 3 + c0].conj();
        tmp1 += (local_src[c1 + 3] - local_src[c1 + 9] * I) *
                local_U[c1 * 3 + c0].conj();
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] -= tmp0 * I;
      local_dest[c0 + 9] += tmp1 * I;
    }
  }
  {
    // forward z
    if (z == local_lat_z - 1) {
      move = 1 - local_lat_z;
    } else {
      move = 1;
    }
    tmp_U = (origin_U + local_lat_tzyxcc * 2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_yxsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] - local_src[c1 + 6] * I) * local_U[c0 * 3 + c1];
        tmp1 +=
            (local_src[c1 + 3] + local_src[c1 + 9] * I) * local_U[c0 * 3 + c1];
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] += tmp0 * I;
      local_dest[c0 + 9] -= tmp1 * I;
    }
  }
  {
    // backward t
    if (t == 0) {
      move = local_lat_t - 1;
    } else {
      move = -1;
    }
    tmp_U = (origin_U + move * local_lat_zyxcc + local_lat_tzyxcc * 3);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_zyxsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
  }
  {
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 +=
            (local_src[c1] + local_src[c1 + 6]) * local_U[c1 * 3 + c0].conj();
        tmp1 += (local_src[c1 + 3] + local_src[c1 + 9]) *
                local_U[c1 * 3 + c0].conj();
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] += tmp0;
      local_dest[c0 + 9] += tmp1;
    }
  }
  {
    // forward t
    if (t == local_lat_t - 1) {
      move = 1 - local_lat_t;
    } else {
      move = 1;
    }
    tmp_U = (origin_U + local_lat_tzyxcc * 3);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = local_U[1] * local_U[5] - local_U[2] * local_U[4];
    local_U[7] = local_U[2] * local_U[3] - local_U[0] * local_U[5];
    local_U[8] = local_U[0] * local_U[4] - local_U[1] * local_U[3];
    norm = sqrt(local_U[6].norm2() * local_U[6].norm2() +
                local_U[7].norm2() * local_U[7].norm2() +
                local_U[8].norm2() * local_U[8].norm2());
    local_U[6] = local_U[6].conj() / norm;
    local_U[7] = local_U[7].conj() / norm;
    local_U[8] = local_U[8].conj() / norm;
    tmp_src = (origin_src + move * local_lat_zyxsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
    for (int c0 = 0; c0 < 3; c0++) {
      tmp0 = zero;
      tmp1 = zero;
      for (int c1 = 0; c1 < 3; c1++) {
        tmp0 += (local_src[c1] - local_src[c1 + 6]) * local_U[c0 * 3 + c1];
        tmp1 += (local_src[c1 + 3] - local_src[c1 + 9]) * local_U[c0 * 3 + c1];
      }
      local_dest[c0] += tmp0;
      local_dest[c0 + 3] += tmp1;
      local_dest[c0 + 6] -= tmp0;
      local_dest[c0 + 9] -= tmp1;
    }
  }
  for (int i = 0; i < 12; i++) {
    dest[i] = local_dest[i];
  }
  // dest = local_dest;
  // memcpy(dest, local_dest, sizeof(local_dest));
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param) {
  int lat_x = param->lattice_size[0];
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  int lat_xcc = lat_x * 9;
  int lat_yxcc = lat_y * lat_xcc;
  int lat_zyxcc = lat_z * lat_yxcc;
  int lat_tzyxcc = lat_t * lat_zyxcc;
  int lat_stzyxcc = 4 * lat_tzyxcc;
  int lat_xsc = lat_x * 12;
  int lat_yxsc = lat_y * lat_xsc;
  int lat_zyxsc = lat_z * lat_yxsc;
  int lat_tzyxsc = lat_t * lat_zyxsc;
  unsigned long gauge_size = lat_stzyxcc * sizeof(LatticeComplex);
  unsigned long fermi_size = lat_tzyxsc * sizeof(LatticeComplex);
  void *device_U;
  void *device_src;
  void *device_dest;
  checkCudaErrors(cudaMalloc(&device_U, gauge_size));
  checkCudaErrors(cudaMalloc(&device_src, fermi_size));
  checkCudaErrors(cudaMalloc(&device_dest, fermi_size));
  checkCudaErrors(
      cudaMemcpy(device_U, gauge, gauge_size, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(device_src, fermion_in, fermi_size, cudaMemcpyHostToDevice));
  dim3 gridSize(lat_x, lat_y, lat_z);
  dim3 blockSize(lat_t);
  clock_t start = clock();
  const int loop(100);
  for (int i = 0; i < loop; i++) {
    dslash<<<gridSize, blockSize>>>(
        device_U, device_src, device_dest, lat_x, lat_y, lat_z, lat_t, lat_xcc,
        lat_yxcc, lat_zyxcc, lat_tzyxcc, lat_xsc, lat_yxsc, lat_zyxsc);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  clock_t end0 = clock();
  checkCudaErrors(
      cudaMemcpy(fermion_out, device_dest, fermi_size, cudaMemcpyDeviceToHost));
  clock_t end1 = clock();
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);
  std::cout << "################"
            << "time cost without cudaMemcpy:"
            << (double)(end0 - start) / loop / CLOCKS_PER_SEC << "s"
            << std::endl;
  std::cout << "################"
            << "time cost with cudaMemcpy:"
            << (double)(end1 - start) / CLOCKS_PER_SEC << "s" << std::endl;
  checkCudaErrors(cudaFree(device_U));
  checkCudaErrors(cudaFree(device_src));
  checkCudaErrors(cudaFree(device_dest));
}
