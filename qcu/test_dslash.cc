#include <filesystem>
#include <limits>
#include <sys/types.h>
#pragma optimize(5)
#include <chrono>
#include <cmath>
#include <cstdio>

#define give_value(U, zero, n)                                                 \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = zero;                                                             \
    }                                                                          \
  }
#define give_ptr(U, origin_U, n)                                               \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] = origin_U[i];                                                      \
    }                                                                          \
  }
#define add(U, tmp, n)                                                         \
  {                                                                            \
    for (int i = 0; i < n; i++) {                                              \
      U[i] += tmp3[i];                                                         \
    }                                                                          \
  }

#define give_u(tmp, tmp_U)                                                     \
  {                                                                            \
    for (int i = 0; i < 6; i++) {                                              \
      tmp[i] = tmp_U[i];                                                       \
    }                                                                          \
    tmp[6] = (tmp[1] * tmp[5] - tmp[2] * tmp[4]).conj();                       \
    tmp[7] = (tmp[2] * tmp[3] - tmp[0] * tmp[5]).conj();                       \
    tmp[8] = (tmp[0] * tmp[4] - tmp[1] * tmp[3]).conj();                       \
  }

#define mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero)                         \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[cc * 3 + c1];                       \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[c0 * 3 + cc] * tmp2[c1 * 3 + cc].conj();                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero)                          \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[cc * 3 + c1];                \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero)                           \
  {                                                                            \
    for (int c0 = 0; c0 < 3; c0++) {                                           \
      for (int c1 = 0; c1 < 3; c1++) {                                         \
        tmp0 = zero;                                                           \
        for (int cc = 0; cc < 3; cc++) {                                       \
          tmp0 += tmp1[cc * 3 + c0].conj() * tmp2[c1 * 3 + cc].conj();         \
        }                                                                      \
        tmp3[c0 * 3 + c1] = tmp0;                                              \
      }                                                                        \
    }                                                                          \
  }

#define inverse(input_matrix, inverse_matrix, augmented_matrix, pivot, factor, \
                size)                                                          \
  {                                                                            \
    for (int i = 0; i < size; ++i) {                                           \
      for (int j = 0; j < size; ++j) {                                         \
        inverse_matrix[i * size + j] = input_matrix[i * size + j];             \
        augmented_matrix[i * 2 * size + j] = inverse_matrix[i * size + j];     \
      }                                                                        \
      augmented_matrix[i * 2 * size + size + i] = 1.0;                         \
    }                                                                          \
    for (int i = 0; i < size; ++i) {                                           \
      pivot = augmented_matrix[i * 2 * size + i];                              \
      for (int j = 0; j < 2 * size; ++j) {                                     \
        augmented_matrix[i * 2 * size + j] /= pivot;                           \
      }                                                                        \
      for (int j = 0; j < size; ++j) {                                         \
        if (j != i) {                                                          \
          factor = augmented_matrix[j * 2 * size + i];                         \
          for (int k = 0; k < 2 * size; ++k) {                                 \
            augmented_matrix[j * 2 * size + k] -=                              \
                factor * augmented_matrix[i * 2 * size + k];                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int i = 0; i < size; ++i) {                                           \
      for (int j = 0; j < size; ++j) {                                         \
        inverse_matrix[i * size + j] =                                         \
            augmented_matrix[i * 2 * size + size + j];                         \
      }                                                                        \
    }                                                                          \
  }

#define print_matrix(input_matrix, size)                                       \
  {                                                                            \
    for (int i = 0; i < size; ++i) {                                           \
      for (int j = 0; j < size; ++j) {                                         \
        printf("(%.d,%.d):(%.16lf,%.16lf)\n", i, j,                            \
               input_matrix[i * size + j].real,                                \
               input_matrix[i * size + j].imag);                               \
      }                                                                        \
    }                                                                          \
  }

#define print_fermi(input_fermi, size)                                         \
  {                                                                            \
    for (int i = 0; i < size; ++i) {                                           \
      printf("(%.d):(%.16lf,%.16lf)\n", i, input_fermi[i].real,                \
             input_fermi[i].imag);                                             \
    }                                                                          \
  }

#define give_rand(input_matrix, size)                                          \
  {                                                                            \
    for (int i = 0; i < size; ++i) {                                           \
      input_matrix[i].real = static_cast<double>(rand()) / RAND_MAX;           \
      input_matrix[i].imag = static_cast<double>(rand()) / RAND_MAX;           \
    }                                                                          \
  }

struct LatticeComplex {
  double real;
  double imag;
  LatticeComplex(const double &real = 0.0, const double &imag = 0.0)
      : real(real), imag(imag) {}
  LatticeComplex &operator=(const LatticeComplex &other) {
    real = other.real;
    imag = other.imag;
    return *this;
  }
  LatticeComplex &operator=(const double &other) {
    real = other;
    imag = 0;
    return *this;
  }
  LatticeComplex operator+(const LatticeComplex &other) const {
    return LatticeComplex(real + other.real, imag + other.imag);
  }
  LatticeComplex operator-(const LatticeComplex &other) const {
    return LatticeComplex(real - other.real, imag - other.imag);
  }
  LatticeComplex operator*(const LatticeComplex &other) const {
    return LatticeComplex(real * other.real - imag * other.imag,
                          real * other.imag + imag * other.real);
  }
  LatticeComplex operator*(const double &other) const {
    return LatticeComplex(real * other, imag * other);
  }
  LatticeComplex operator/(const LatticeComplex &other) const {
    double denom = other.real * other.real + other.imag * other.imag;
    return LatticeComplex((real * other.real + imag * other.imag) / denom,
                          (imag * other.real - real * other.imag) / denom);
  }
  LatticeComplex operator/(const double &other) const {
    return LatticeComplex(real / other, imag / other);
  }
  LatticeComplex operator-() const { return LatticeComplex(-real, -imag); }
  bool operator==(const LatticeComplex &other) const {
    return (real == other.real && imag == other.imag);
  }
  bool operator!=(const LatticeComplex &other) const {
    return !(*this == other);
  }
  LatticeComplex &operator+=(const LatticeComplex &other) {
    real = real + other.real;
    imag = imag + other.imag;
    return *this;
  }
  LatticeComplex &operator-=(const LatticeComplex &other) {
    real = real - other.real;
    imag = imag - other.imag;
    return *this;
  }
  LatticeComplex &operator*=(const LatticeComplex &other) {
    real = real * other.real - imag * other.imag;
    imag = real * other.imag + imag * other.real;
    return *this;
  }
  LatticeComplex &operator*=(const double &other) {
    real = real * other;
    imag = imag * other;
    return *this;
  }
  LatticeComplex &operator/=(const LatticeComplex &other) {
    double denom = other.real * other.real + other.imag * other.imag;
    real = (real * other.real + imag * other.imag) / denom;
    imag = (imag * other.real - real * other.imag) / denom;
    return *this;
  }
  LatticeComplex &operator/=(const double &other) {
    real = real / other;
    imag = imag / other;
    return *this;
  }
  LatticeComplex conj() const { return LatticeComplex(real, -imag); }
  double norm2() const { return sqrt(real * real + imag * imag); }
};

void dslash(void *device_U, void *device_src, void *device_dest,
            int device_lat_x, const int device_lat_y, const int device_lat_z,
            const int device_lat_t, const int device_parity) {
  int parity;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int x = 0;
  const int y = 0;
  const int z = 0;
  const int t = 0;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  const int lat_xsc = lat_x * 12;
  const int lat_yxsc = lat_y * lat_xsc;
  const int lat_zyxsc = lat_z * lat_yxsc;
  int move;
  parity = device_parity;
  const int oe = (y + z + t) % 2;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_src =
      ((static_cast<LatticeComplex *>(device_src)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) + t * lat_zyxsc +
       z * lat_yxsc + y * lat_xsc + x * 12);
  LatticeComplex *tmp_U;
  LatticeComplex *tmp_src;
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex tmp1(0.0, 0.0);
  LatticeComplex U[9];
  LatticeComplex src[12];
  LatticeComplex dest[12];
  // print_fermi(origin_U, 9);
  // print_fermi(origin_src, 12);
  // just wilson(Sum part)
  give_value(dest, zero, 12);
  {
    // x-1
    move = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move * 9 + (1 - parity) * lat_tzyxcc);
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * 12);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_xsc);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_yxsc);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
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
    give_u(U, tmp_U);
    tmp_src = (origin_src + move * lat_zyxsc);
    give_ptr(src, tmp_src, 12);
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
  give_ptr(origin_dest, dest, 12);
  // print_fermi(dest, 12);
}

void clover(void *device_U, void *device_clover, int device_lat_x,
            const int device_lat_y, const int device_lat_z,
            const int device_lat_t, const int device_parity) {
  int parity;
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int lat_t = device_lat_t;
  const int lat_xcc = lat_x * 9;
  const int lat_yxcc = lat_y * lat_xcc;
  const int lat_zyxcc = lat_z * lat_yxcc;
  const int lat_tzyxcc = lat_t * lat_zyxcc;
  int move0;
  int move1;
  const int x = 0;
  const int y = 0;
  const int z = 0;
  const int t = 0;
  const int oe = (y + z + t) % 2;
  LatticeComplex I(0.0, 1.0);
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex tmp0(0.0, 0.0);
  LatticeComplex *origin_U =
      ((static_cast<LatticeComplex *>(device_U)) + t * lat_zyxcc +
       z * lat_yxcc + y * lat_xcc + x * 9);
  LatticeComplex *origin_clover =
      ((static_cast<LatticeComplex *>(device_clover)) + t * lat_zyxcc * 16 +
       z * lat_yxcc * 16 + y * lat_xcc * 16 + x * 144);
  LatticeComplex *tmp_U;
  LatticeComplex tmp1[9];
  LatticeComplex tmp2[9];
  LatticeComplex tmp3[9];
  LatticeComplex U[9];
  LatticeComplex clover[144];
  // sigmaF
  {
    parity = device_parity;
    give_value(clover, zero, 144);
    give_value(origin_clover, zero, 144);
    give_value(tmp1, zero, 9);
    give_value(tmp2, zero, 9);
  }
  // XY
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    // print_fermi(tmp_U, 9);

    //// x+1,y,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    // print_fermi(tmp_U, 9);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y+1,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 2 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y-1,z,t;y;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t;x
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_xcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z+1,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 4 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z-1,t;z;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;x
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
  give_value(U, zero, 9);
  {
    //// x,y,z,t;x
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x+1,y,z,t;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe != parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t+1;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + lat_tzyxcc * 6 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x-1,y,z,t;x;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe == parity);
    tmp_U = (origin_U + move0 * 9 + (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x-1,y,z,t-1;t;dag
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move0 = (-1 + (x == 0) * lat_x) * (oe != parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;x
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move0 = (1 - (x == lat_x - 1) * lat_x) * (oe == parity);
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * 9 + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    move0 = 0;
    tmp_U = (origin_U + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z+1,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z-1,t;z;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t;y
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_yxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
  give_value(U, zero, 9);
  {
    //// x,y,z,t;y
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y+1,z,t;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t+1;y;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y-1,z,t;y;dag
    move0 = -1 + (y == 0) * lat_y;
    tmp_U = (origin_U + move0 * lat_xcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y-1,z,t-1;t;dag
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move0 = -1 + (y == 0) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 2 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;y
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 2 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move0 = 1 - (y == lat_y - 1) * lat_y;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_xcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 2 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
  give_value(U, zero, 9);
  {
    //// x,y,z,t;z
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z+1,t;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t;t
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 6 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t+1;z;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = 1 - (t == lat_t - 1) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z-1,t;z;dag
    move0 = -1 + (z == 0) * lat_z;
    tmp_U = (origin_U + move0 * lat_yxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z-1,t-1;t;dag
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move0 = -1 + (z == 0) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 4 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
  {
    //// x,y,z,t-1;t;dag
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 6 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    //// x,y,z,t-1;z
    move0 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_zyxcc + lat_tzyxcc * 4 +
             (1 - parity) * lat_tzyxcc);
    give_u(tmp2, tmp_U);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move0 = 1 - (z == lat_z - 1) * lat_z;
    move1 = -1 + (t == 0) * lat_t;
    tmp_U = (origin_U + move0 * lat_yxcc + move1 * lat_zyxcc + lat_tzyxcc * 6 +
             parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    move0 = 0;
    tmp_U = (origin_U + lat_tzyxcc * 4 + parity * lat_tzyxcc);
    give_u(tmp1, tmp_U);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add(U, tmp3, 9);
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
    // A=1+T
    int j = 13;
    LatticeComplex one(1.0, 0);
    for (int i = 0; i < 144; i++) {
      if (j == 13) {
        clover[i] += one;
        j = 0;
      }
      j++;
      origin_clover[i] = clover[i];
    }
  }
  // print_matrix(clover, 12);
  // print_matrix(U, 3);
}

void give_clover(void *device_propagator, void *device_dest, int device_lat_x,
                 const int device_lat_y, const int device_lat_z) {
  const int lat_x = device_lat_x;
  const int lat_y = device_lat_y;
  const int lat_z = device_lat_z;
  const int x = 0;
  const int y = 0;
  const int z = 0;
  const int t = 0;
  LatticeComplex zero(0.0, 0.0);
  LatticeComplex I(0.0, 1.0);
  LatticeComplex tmp0(0.0, 0.0);
  int tmp1;
  int tmp2;
  LatticeComplex *origin_propagator =
      ((static_cast<LatticeComplex *>(device_propagator)) +
       t * lat_z * lat_y * lat_x * 144 + z * lat_y * lat_x * 144 +
       y * lat_x * 144 + x * 144);
  LatticeComplex *origin_dest =
      ((static_cast<LatticeComplex *>(device_dest)) +
       t * lat_z * lat_y * lat_x * 12 + z * lat_y * lat_x * 12 +
       y * lat_x * 12 + x * 12);
  LatticeComplex input_propagator[144];
  LatticeComplex propagator[144];
  LatticeComplex augmented_propagator[288];
  LatticeComplex pivot;
  LatticeComplex factor;
  LatticeComplex dest[12];
  LatticeComplex tmp[12];
  // print_fermi(origin_propagator, 144);
  printf("###########################\n");
  give_ptr(input_propagator, origin_propagator, 144);
  give_value(augmented_propagator, zero, 288);
  // give_ptr(dest, origin_dest, 12);
  inverse(input_propagator, propagator, augmented_propagator, pivot, factor,
          12);
  // print_fermi(propagator, 144);
  give_ptr(input_propagator, propagator, 144);
  give_value(tmp, zero, 12);
  {
    for (int sc0 = 0; sc0 < 12; sc0++) {
      for (int sc1 = 0; sc1 < 12; sc1++) {
        tmp0 = zero;
        for (int scsc = 0; scsc < 12; scsc++) {
          tmp0 += input_propagator[sc0 * 12 + scsc] *
                  origin_propagator[scsc * 12 + sc1];
        }
        printf("(%.d,%.d):(%.16lf,%.16lf)\n", sc0, sc1, tmp0.real, tmp0.imag);
        propagator[sc0 * 12 + sc1] = tmp0;
      }
    }
  }
  {
    for (int sc0 = 0; sc0 < 12; sc0++) {
      tmp0 = zero;
      for (int sc1 = 0; sc1 < 12; sc1++) {
        tmp0 += propagator[sc0 * 12 + sc1] * dest[sc1];
      }
      tmp[sc0] = tmp0;
    }
    give_ptr(origin_dest, tmp, 12);
  }
}

int main() {
  int lat_x = 1;
  int lat_y = 1;
  int lat_z = 1;
  int lat_t = 1;
  int parity = 1;
  LatticeComplex gauge[18 * 4];
  LatticeComplex fermion_in[12];
  LatticeComplex fermion_out[12];
  LatticeComplex propagator[144];
  {
    give_rand(gauge, 18 * 4);
    give_rand(fermion_in, 12);
    // print_fermi(gauge, 18);
    // print_fermi(fermion_in, 12);
  }
  {
    // wilson dslash
    auto start = std::chrono::high_resolution_clock::now();
    dslash(gauge, fermion_in, fermion_out, lat_x, lat_y, lat_z, lat_t, parity);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "wilson dslash total time: (without malloc free memcpy) : %.16lf sec\n",
        double(duration) / 1e9);
    // print_fermi(fermion_out, 12);
  }
  {
    // just clover
    auto start = std::chrono::high_resolution_clock::now();
    clover(gauge, (void *)propagator, lat_x, lat_y, lat_z, lat_t, parity);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "just clover total time: (without malloc free memcpy) :%.16lf sec\n ",
        double(duration) / 1e9);
  }
  {
    // give clover
    auto start = std::chrono::high_resolution_clock::now();
    give_clover((void *)propagator, fermion_out, lat_x, lat_y, lat_z);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf(
        "give clover total time: (without malloc free memcpy) :%.16lf sec\n ",
        double(duration) / 1e9);
  }
  return 0;
}
