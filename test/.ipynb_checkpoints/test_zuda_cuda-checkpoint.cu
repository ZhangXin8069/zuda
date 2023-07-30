#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// 定义矩阵的维度
const int M = 2048;
const int N = 2048;
const int K = 2048;

// 定义矩阵的类型
typedef float Real;

// 定义矩阵的类
class Matrix {
 public:
  Real *data;
  int rows;
  int cols;

  Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = (Real *)malloc(sizeof(Real) * rows * cols);
  }

  ~Matrix() {
    free(this->data);
  }

  void set(int row, int col, Real value) {
    this->data[row * this->cols + col] = value;
  }

  Real get(int row, int col) {
    return this->data[row * this->cols + col];
  }

  void print() {
    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->cols; j++) {
        printf("%f ", this->data[i * this->cols + j]);
      }
      printf("\n");
    }
  }

  // 将矩阵转换为一个数组
  float *to_array() {
    float *array = (float *)malloc(sizeof(float) * this->rows * this->cols);
    for (int i = 0; i < this->rows; i++) {
      for (int j = 0; j < this->cols; j++) {
        array[i * this->cols + j] = this->data[i * this->cols + j];
      }
    }
    return array;
  }
};

// 定义矩阵乘法函数
__global__ void matrix_multiplication(Matrix *A, Matrix *B, Matrix *C) {
  // 获取线程的索引
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // 计算矩阵乘法
  Real sum = 0.0;
  for (int k = 0; k < A->cols; k++) {
    sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
  }

  // 将结果存储到矩阵 C 中
  C->data[i * C->cols + j] = sum;
}

// 主函数
int main() {
  // 创建两个矩阵 A 和 B
  Matrix A(M, K);
  Matrix B(K, N);

  // 初始化矩阵 A 和 B
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A.set(i, j, i + j);
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B.set(i, j, i + j);
    }
  }

  // 创建矩阵 C
  Matrix C(M, N);

  // 将矩阵 A 和 B 从主机复制到设备上
  cudaMemcpyToSymbol("A", &A, sizeof(Matrix));
  cudaMemcpyToSymbol("B", &B, sizeof(Matrix));

  // 启动矩阵乘法函数
  dim3 block(32, 32);
  dim3 grid(M / block.x, N / block.y);
  matrix_multiplication<<<grid, block>>>(&A, &B, &C);

  // 将矩阵 C 从设备复制到主机上
  cudaMemcpyFromSymbol("C", &C, sizeof(Matrix));

  // 打印矩阵 C
  return 0;
}