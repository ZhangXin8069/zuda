#include <chrono>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;

typedef complex<double> Complex;

// 打印矩阵
void printMatrix(const Complex *inputMatrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cout << inputMatrix[i * cols + j] << "\t";
    }
    cout << endl;
  }
}

// 高斯-约旦法求逆矩阵
void gaussJordanInverse(const Complex *inputMatrix, Complex *inverseMatrix,
                        int size) {
  // 构造增广矩阵
  Complex augmentedMatrix[2 * size * size];
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      inverseMatrix[i * size + j] = inputMatrix[i * size + j];
      augmentedMatrix[i * 2 * size + j] = inverseMatrix[i * size + j];
    }
    augmentedMatrix[i * 2 * size + size + i] = 1.0;
  }

  // 高斯消元
  for (int i = 0; i < size; ++i) {
    Complex pivot = augmentedMatrix[i * 2 * size + i];
    for (int j = 0; j < 2 * size; ++j) {
      augmentedMatrix[i * 2 * size + j] /= pivot;
    }
    for (int j = 0; j < size; ++j) {
      if (j != i) {
        Complex factor = augmentedMatrix[j * 2 * size + i];
        for (int k = 0; k < 2 * size; ++k) {
          augmentedMatrix[j * 2 * size + k] -=
              factor * augmentedMatrix[i * 2 * size + k];
        }
      }
    }
  }

  // 提取逆矩阵
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      inverseMatrix[i * size + j] = augmentedMatrix[i * 2 * size + size + j];
    }
  }
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

int main() {
  srand(time(0));

  int size = 12; // 矩阵大小为3x3
  Complex inputMatrix[size * size];
  Complex inverseMatrix[size * size];
  Complex augmentedMatrixMatrix[size * size * 2];
  Complex pivot;
  Complex factor;

  // 生成随机复数矩阵
  for (int i = 0; i < size * size; ++i) {
    double realPart = static_cast<double>(rand()) / RAND_MAX;
    double imagPart = static_cast<double>(rand()) / RAND_MAX;
    inputMatrix[i] = Complex(realPart, imagPart);
  }

  cout << "Input Matrix:" << endl;
  printMatrix(inputMatrix, size, size);

  // 求逆矩阵
  auto start = std::chrono::high_resolution_clock::now();
  // gaussJordanInverse(inputMatrix, inverseMatrix, size);
  inverse(inputMatrix, inverseMatrix, augmentedMatrixMatrix, pivot, factor,
          size);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("gaussJordanInverse total time: (without malloc "
         "free memcpy) : "
         "%.9lf "
         "sec\n",
         double(duration) / 1e9);

  cout << "\nInverse Matrix:" << endl;
  printMatrix(inverseMatrix, size, size);

  // 验证
  cout << "\nVerification:" << endl;
  Complex resultMatrix[size * size];
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      Complex sum = 0;
      for (int k = 0; k < size; ++k) {
        sum += inputMatrix[i * size + k] * inverseMatrix[k * size + j];
      }
      resultMatrix[i * size + j] = sum;
    }
  }
  printMatrix(resultMatrix, size, size);

  return 0;
}
