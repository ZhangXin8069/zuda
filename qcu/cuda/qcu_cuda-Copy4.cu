#pragma nv_verbose
#pragma optimize(5)
#include "qcu.h"
#include <assert.h>
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
  __device__ LatticeComplex(const double &real = 0.0, const double &imag = 0.0)
      : real(real), imag(imag) {}
  __device__ LatticeComplex &operator=(const LatticeComplex &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  __device__ LatticeComplex &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  __device__ LatticeComplex operator+(const LatticeComplex &other) const {
    return LatticeComplex(real + other.real, imag + other.imag);
  }
  __device__ LatticeComplex operator-(const LatticeComplex &other) const {
    return LatticeComplex(real - other.real, imag - other.imag);
  }
  __device__ LatticeComplex operator*(const LatticeComplex &other) const {
    return LatticeComplex(real * other.real - imag * other.imag,
                          real * other.imag + imag * other.real);
  }
  __device__ LatticeComplex operator*(const double &other) const {
    return LatticeComplex(real * other, imag * other);
  }
  __device__ LatticeComplex operator/(const LatticeComplex &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeComplex((real * other.real + imag * other.imag) / denom,
                          (imag * other.real - real * other.imag) / denom);
  }
  __device__ LatticeComplex operator/(const double &other) const {
    return LatticeComplex(real / other, imag / other);
  }
  __device__ LatticeComplex operator-() const {
    return LatticeComplex(-real, -imag);
  }
  __device__ bool operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  __device__ bool operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  __device__ LatticeComplex &operator+=(const LatticeComplex &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  __device__ LatticeComplex &operator-=(const LatticeComplex &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  __device__ LatticeComplex &operator*=(const LatticeComplex &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  __device__ LatticeComplex &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  __device__ LatticeComplex &operator/=(const LatticeComplex &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  __device__ LatticeComplex &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  __device__ LatticeComplex conj() const { return LatticeComplex(real, -imag); }
  __device__ double norm2() const { return sqrt(real * real + imag * imag); }
};
__global__ void dslash(void *device_U, void *device_src, void *device_dest,
                       int lat_x, const int lat_y, const int lat_z,
                       const int lat_t, const int lat_xcc, int lat_yxcc,
                       const int lat_zyxcc, const int lat_tzyxcc,
                       const int lat_xsc, int lat_yxsc, const int lat_zyxsc,
                       const int parity) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int t = thread / (lat_x * lat_y * lat_z);
  thread -= t * (lat_x * lat_y * lat_z);
  int z = thread / (lat_x * lat_y);
  thread -= z * (lat_x * lat_y);
  int y = thread / lat_x;
  int x = thread - y * lat_x;
  int move;
  int eo = (y + z + t) % 2;
  int local_lat_x = lat_x;
  int local_lat_y = lat_y;
  int local_lat_z = lat_z;
  int local_lat_t = lat_t;
  int local_lat_xsc = lat_xsc;
  int local_lat_yxsc = lat_yxsc;
  int local_lat_zyxsc = lat_zyxsc;
  int local_lat_xcc = lat_xcc;
  int local_lat_yxcc = lat_yxcc;
  int local_lat_zyxcc = lat_zyxcc;
  int local_lat_tzyxcc = lat_tzyxcc;
  int local_lat_tzyxcc2 = local_lat_tzyxcc * 2;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * local_lat_zyxcc +
       z * local_lat_yxcc + y * local_lat_xcc + x * 9 +
       parity * local_lat_tzyxcc);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * local_lat_zyxsc +
       z * local_lat_yxsc + y * local_lat_xsc + x * 12);
  LatticeComplex *dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * local_lat_zyxsc +
       z * local_lat_yxsc + y * local_lat_xsc + x * 12);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex local_U[9];
  LatticeComplex local_src[12];
  LatticeComplex local_dest[12];
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
    // if (x == 0) {
    //   move = local_lat_x - 1;
    // } else {
    //   move = -1;
    // }
    move = (1 & ~(parity ^ eo)) * (-1 + (x == 0) * local_lat_x + x) +
           (parity ^ eo) * x - x;
    tmp_U = (origin_U + move * 9);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // if (x == local_lat_x - 1) {
    //   move = 1 - local_lat_x;
    // } else {
    //   move = 1;
    // }
    move = (1 & ~(parity ^ eo)) * x +
           (parity ^ eo) * (1 - (x == local_lat_x - 1) * local_lat_x + x) - x;
    tmp_U = origin_U;
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // if (y == 0) {
    //   move = local_lat_y - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (y == 0) * local_lat_y;
    tmp_U = (origin_U + move * local_lat_xcc + local_lat_tzyxcc2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // // forward y
    // if (y == local_lat_y - 1) {
    //   move = 1 - local_lat_y;
    // } else {
    //   move = 1;
    // }
    move = 1 - (y == local_lat_y - 1) * local_lat_y;
    tmp_U = (origin_U + local_lat_tzyxcc2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // if (z == 0) {
    //   move = local_lat_z - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (z == 0) * local_lat_z;
    tmp_U = (origin_U + move * local_lat_yxcc + local_lat_tzyxcc2 * 2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // if (z == local_lat_z - 1) {
    //   move = 1 - local_lat_z;
    // } else {
    //   move = 1;
    // }
    move = 1 - (z == local_lat_z - 1) * local_lat_z;
    tmp_U = (origin_U + local_lat_tzyxcc2 * 2);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
    // if (t == 0) {
    //   move = local_lat_t - 1;
    // } else {
    //   move = -1;
    // }
    move = -1 + (t == 0) * local_lat_t;
    tmp_U = (origin_U + move * local_lat_zyxcc + local_lat_tzyxcc2 * 3);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
    tmp_src = (origin_src + move * local_lat_zyxsc);
    for (int i = 0; i < 12; i++) {
      local_src[i] = tmp_src[i];
    }
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
    // if (t == local_lat_t - 1) {
    //   move = 1 - local_lat_t;
    // } else {
    //   move = 1;
    // }
    move = 1 - (t == local_lat_t - 1) * local_lat_t;
    tmp_U = (origin_U + local_lat_tzyxcc2 * 3);
    for (int i = 0; i < 6; i++) {
      local_U[i] = tmp_U[i];
    }
    local_U[6] = (local_U[1] * local_U[5] - local_U[2] * local_U[4]).conj();
    local_U[7] = (local_U[2] * local_U[3] - local_U[0] * local_U[5]).conj();
    local_U[8] = (local_U[0] * local_U[4] - local_U[1] * local_U[3]).conj();
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
}

void dslashQcu(void *fermion_out, void *fermion_in, void *gauge,
               QcuParam *param, int parity) {
  int lat_x = param->lattice_size[0] >> 1;
  int lat_y = param->lattice_size[1];
  int lat_z = param->lattice_size[2];
  int lat_t = param->lattice_size[3];
  int lat_xcc = lat_x * 9;
  int lat_yxcc = lat_y * lat_xcc;
  int lat_zyxcc = lat_z * lat_yxcc;
  int lat_tzyxcc = lat_t * lat_zyxcc;
  int lat_xsc = lat_x * 12;
  int lat_yxsc = lat_y * lat_xsc;
  int lat_zyxsc = lat_z * lat_yxsc;
  dim3 gridDim(lat_tzyxcc / 9 / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE);
  checkCudaErrors(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();
  dslash<<<gridDim, blockDim>>>(
      gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, lat_xcc,
      lat_yxcc, lat_zyxcc, lat_tzyxcc, lat_xsc, lat_yxsc, lat_zyxsc, parity);
  cudaError_t err = cudaGetLastError();
  checkCudaErrors(err);
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("total time: (without malloc free memcpy) : %.9lf sec\n",
         double(duration) / 1e9);
}