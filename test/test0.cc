#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

typedef complex<double> Complex;

// 打印矩阵
void printMatrix(const vector<Complex> &matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cout << matrix[i * cols + j] << "\t";
    }
    cout << endl;
  }
}

// 高斯-约旦法求逆矩阵
void gaussJordanInverse(vector<Complex> &matrix, int size) {
  // 构造增广矩阵
  vector<vector<Complex>> augmentedMatrix(size, vector<Complex>(2 * size));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      augmentedMatrix[i * 2 * size + j] = matrix[i * size + j];
    }
    augmentedMatrix[i * 2 * size + i + size] = 1.0;
  }

  // 高斯消元
  for (int i = 0; i < size; ++i) {
    Complex pivot = augmentedMatrix[i][i];
    for (int j = 0; j < 2 * size; ++j) {
      augmentedMatrix[i * 2 * size + j] /= pivot;
    }
    for (int j = 0; j < size; ++j) {
      if (j != i) {
        Complex factor = augmentedMatrix[j][i];
        for (int k = 0; k < 2 * size; ++k) {
          augmentedMatrix[i * 2 * size + k] -= factor * augmentedMatrix[i][k];
        }
      }
    }
  }

  // 提取逆矩阵
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i * size + j] = augmentedMatrix[i][size + j];
    }
  }
}

int main() {
  srand(time(0));

  int size = 6; // 矩阵大小为3x3
  vector<Complex> inputMatrix(size * size);

  // 生成随机复数矩阵
  for (int i = 0; i < size * size; ++i) {
    double realPart = static_cast<double>(rand()) / RAND_MAX;
    double imagPart = static_cast<double>(rand()) / RAND_MAX;
    inputMatrix[i] = Complex(realPart, imagPart);
  }

  cout << "Input Matrix:" << endl;
  printMatrix(inputMatrix, size, size);

  // 求逆矩阵
  vector<Complex> inverseMatrix = inputMatrix; // 复制输入矩阵
  gaussJordanInverse(inverseMatrix, size);

  cout << "\nInverse Matrix:" << endl;
  printMatrix(inverseMatrix, size, size);

  // 验证
  cout << "\nVerification:" << endl;
  vector<Complex> resultMatrix(size * size);
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
