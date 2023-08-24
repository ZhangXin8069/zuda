#include <cmath>
#include <iostream>
using namespace std;
const int N = 3;
double a[N][N], b[N][N], L[N][N];
int main() {
  // 随机生成一个3x3的对称正定矩阵
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      a[i][j] = rand() % 10 + 1;
      a[j][i] = a[i][j];
    }
    a[i][i] += N;
  }
  // 输出原始矩阵
  cout << "Original matrix:" << endl;
  for (int i = 0; i < N; i++, puts(""))
    for (int j = 0; j < N; j++)
      printf("%.2lf ", a[i][j]);
  // 初始化单位矩阵
  for (int i = 0; i < N; i++)
    b[i][i] = 1;
  // 进行Cholesky分解
  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < j; k++)
        sum += L[i][k] * L[j][k];
      if (i == j)
        L[i][j] = sqrt(a[i][i] - sum);
      else
        L[i][j] = (a[i][j] - sum) / L[j][j];
    }
  }
  // 求逆
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = b[i][j];
      for (int k = i - 1; k >= 0; k--)
        sum -= L[i][k] * b[k][j];
      b[i][j] = sum / L[i][i];
    }
  }
  for (int i = N - 1; i >= 0; i--) {
    for (int j = 0; j < N; j++) {
      double sum = b[i][j];
      for (int k = i + 1; k < N; k++)
        sum -= L[k][i] * b[k][j];
      b[i][j] = sum / L[i][i];
    }
  }
  // 输出逆矩阵
  cout << "Inverse matrix:" << endl;
  for (int i = 0; i < N; i++, puts(""))
    for (int j = 0; j < N; j++)
      printf("%.2lf ", b[j][i]);

  // 验证结果
  cout << "A * A^(-1):" << endl;
  double c[N][N];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = 0;
      for (int k = 0; k < N; k++)
        c[i][j] += a[i][k] * b[k][j];
      printf("%.2lf ", c[i][j]);
    }
    puts("");
  }

  return 0;
}
