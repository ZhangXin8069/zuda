#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <random>
#include <mpi.h>
// #include "../include/zuda_cpu.h"
// #include "../include/zuda_cuda.h"
class Complex
{
public:
    // double data[2];
    double real;
    double imag;
    __host__ __device__ Complex(double real = 0.0, double imag = 0.0)
    {
        this->real = real;
        this->imag = imag;
    }
    __host__ __device__ Complex &operator=(const Complex &other)
    {
        if (this != &other)
        {
            this->real = other.real;
            this->imag = other.imag;
        }
        return *this;
    }
    __host__ __device__ Complex operator+(const Complex &other) const
    {
        return Complex(this->real + other.real, this->imag + other.imag);
    }
    __host__ __device__ Complex operator-(const Complex &other) const
    {
        return Complex(this->real - other.real, this->imag - other.imag);
    }
    __host__ __device__ Complex operator*(const Complex &other) const
    {
        return Complex(this->real * other.real - this->imag * other.imag,
                       this->real * other.imag + this->imag * other.real);
    }
    __host__ __device__ Complex operator*(const double &other) const
    {
        return Complex(this->real * other, this->imag * other);
    }
    __host__ __device__ Complex operator/(const Complex &other) const
    {
        double denom = other.real * other.real + other.imag * other.imag;
        return Complex((this->real * other.real + this->imag * other.imag) / denom,
                       (this->imag * other.real - this->real * other.imag) / denom);
    }
    __host__ __device__ Complex operator/(const double &other) const
    {
        return Complex(this->real / other, this->imag / other);
    }
    __host__ __device__ Complex operator-() const
    {
        return Complex(-this->real, -this->imag);
    }
    __host__ __device__ Complex &operator+=(const Complex &other)
    {
        this->real += other.real;
        this->imag += other.imag;
        return *this;
    }
    __host__ __device__ Complex &operator-=(const Complex &other)
    {
        this->real -= other.real;
        this->imag -= other.imag;
        return *this;
    }
    __host__ __device__ Complex &operator*=(const Complex &other)
    {
        this->real = this->real * other.real - this->imag * other.imag;
        this->imag = this->real * other.imag + this->imag * other.real;
        return *this;
    }
    __host__ __device__ Complex &operator*=(const double &scalar)
    {
        this->real *= scalar;
        this->imag *= scalar;
        return *this;
    }
    __host__ __device__ Complex &operator/=(const Complex &other)
    {
        double denom = other.real * other.real + other.imag * other.imag;
        this->real = (real * other.real + imag * other.imag) / denom;
        this->imag = (imag * other.real - real * other.imag) / denom;
        return *this;
    }
    __host__ __device__ Complex &operator/=(const double &other)
    {
        this->real /= other;
        this->imag /= other;
        return *this;
    }
    __host__ __device__ bool operator==(const Complex &other) const
    {
        return (this->real == other.real && this->imag == other.imag);
    }
    __host__ __device__ bool operator!=(const Complex &other) const
    {
        return !(*this == other);
    }
    __host__ friend std::ostream &operator<<(std::ostream &os, const Complex &c)
    {
        if (c.imag >= 0.0)
        {
            os << c.real << " + " << c.imag << "i";
        }
        else
        {
            os << c.real << " - " << std::abs(c.imag) << "i";
        }
        return os;
    }
    __host__ __device__ Complex conj()
    {
        return Complex(this->real, -this->imag);
    }
};

class LatticeFermi
{
public:
    int lat_x, lat_y, lat_z, lat_t, lat_s, lat_c;
    int size;
    Complex *lattice_vec;
    __host__ __device__ LatticeFermi(const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c)
        : lat_x(lat_x), lat_y(lat_y), lat_z(lat_z), lat_t(lat_t), lat_s(lat_s), lat_c(lat_c), size(lat_x * lat_y * lat_z * lat_t * lat_s * lat_c)
    {
        this->lattice_vec = new Complex[size];
    }
    __host__ __device__ ~LatticeFermi()
    {
        if (this->lattice_vec != nullptr)
        {
            this->lattice_vec = nullptr;
            delete[] this->lattice_vec;
        }
    }
    __host__ __device__ LatticeFermi &operator=(const LatticeFermi &other)
    {
        if (this != &other)
        {
            this->lat_x = other.lat_x;
            this->lat_y = other.lat_y;
            this->lat_z = other.lat_z;
            this->lat_t = other.lat_t;
            this->lat_s = other.lat_s;
            this->lat_c = other.lat_c;
            this->size = other.size;
            delete[] this->lattice_vec;
            this->lattice_vec = new Complex[size];
            for (int i = 0; i < this->size; i++)
            {
                this->lattice_vec[i] = other.lattice_vec[i];
            }
        }
        return *this;
    }
    __host__ __device__ void assign_zero()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = 0;
            this->lattice_vec[i].imag = 0;
        }
    }
    __host__ __device__ void assign_unit()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = 1;
            this->lattice_vec[i].imag = 0;
        }
    }
    __host__ void assign_random(unsigned seed = 32767)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = u(e);
            this->lattice_vec[i].imag = u(e);
        }
    }
    __host__ __device__ const Complex &operator[](const int &index) const
    {
        return this->lattice_vec[index];
    }
    __host__ __device__ Complex &operator[](const int &index)
    {
        return this->lattice_vec[index];
    }
    __host__ __device__ const Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c) const
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_z * this->lat_t * this->lat_s * this->lat_c + index_t * this->lat_s * this->lat_c + index_s * this->lat_c + index_c;
        return this->lattice_vec[index];
    }
    __host__ __device__ Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_z * this->lat_t * this->lat_s * this->lat_c + index_t * this->lat_s * this->lat_c + index_s * this->lat_c + index_c;
        return this->lattice_vec[index];
    }
    __host__ __device__ LatticeFermi operator+(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] + other[i];
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator-(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] - other[i];
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator-() const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = -this->lattice_vec[i];
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator*(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] * other[i];
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator/(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] / other[i];
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator+(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] + other;
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator-(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] - other;
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator*(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] * other;
        }
        return result;
    }
    __host__ __device__ LatticeFermi operator/(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] / other;
        }
        return result;
    }
    __host__ __device__ bool operator==(const LatticeFermi &other) const
    {
        if (this->size != other.size)
        {
            return false;
        }
        for (int i = 0; i < this->size; ++i)
        {
            if (this->lattice_vec[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }
    __host__ __device__ bool operator!=(const LatticeFermi &other) const
    {
        return !(*this == other);
    }
    __host__ __device__ void print(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
    {
    }
    __host__ __device__ void print()
    {
        for (int x = 0; x < this->lat_x; x++)
        {
            for (int y = 0; y < this->lat_y; y++)
            {
                for (int z = 0; z < this->lat_z; z++)
                {
                    for (int t = 0; t < this->lat_t; t++)
                    {
                        for (int s = 0; s < this->lat_s; s++)
                        {
                            for (int c = 0; c < this->lat_c; c++)
                            {
                                print(x, y, z, t, s, c);
                            }
                        }
                    }
                }
            }
        }
    }
    __host__ __device__ double norm_2()
    {
        double result = 0;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].real * this->lattice_vec[i].real + this->lattice_vec[i].imag * this->lattice_vec[i].imag;
        }
        return result;
    }
    __host__ double norm_2X()
    {
        double local_result = 0;
        double global_result = 0;
        local_result = norm_2();
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_result;
    }
    __host__ __device__ Complex dot(const LatticeFermi &other)
    {
        Complex result;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].conj() * other[i];
        }
        return result;
    }
    __host__ Complex dotX(const LatticeFermi &other)
    {
        Complex local_result;
        Complex global_result;
        local_result = dot(other);
        MPI_Allreduce(&local_result, &global_result, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return global_result;
    }
};
class Gamme
{
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

class LatticeGauge
{
public:
    int lat_x, lat_y, lat_z, lat_t, lat_s, lat_c0, lat_c1;
    int size;
    Complex *lattice_vec;
    __host__ __device__ LatticeGauge(const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c)
        : lat_x(lat_x), lat_y(lat_y), lat_z(lat_z), lat_t(lat_t), lat_s(lat_s), lat_c0(lat_c), lat_c1(lat_c), size(lat_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1)
    {
        this->lattice_vec = new Complex[size];
    }
    __host__ __device__ ~LatticeGauge()
    {
        if (this->lattice_vec != nullptr)
        {
            this->lattice_vec = nullptr;
            delete[] this->lattice_vec;
        }
    }
    __host__ __device__ LatticeGauge &operator=(const LatticeGauge &other)
    {
        if (this != &other)
        {
            this->lat_x = other.lat_x;
            this->lat_y = other.lat_y;
            this->lat_z = other.lat_z;
            this->lat_t = other.lat_t;
            this->lat_s = other.lat_s;
            this->lat_c0 = other.lat_c0;
            this->lat_c1 = other.lat_c1;
            this->size = other.size;
            delete[] this->lattice_vec;
            this->lattice_vec = new Complex[size];
            for (int i = 0; i < this->size; i++)
            {
                this->lattice_vec[i] = other.lattice_vec[i];
            }
        }
        return *this;
    }
    __host__ __device__ void assign_zero()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = 0;
            this->lattice_vec[i].imag = 0;
        }
    }
    __host__ __device__ void assign_unit()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = 1;
            this->lattice_vec[i].imag = 0;
        }
    }
    __host__ void assign_random(unsigned seed = 32767)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].real = u(e);
            this->lattice_vec[i].imag = u(e);
        }
    }
    __host__ __device__ const Complex &operator[](const int &index) const
    {
        return this->lattice_vec[index];
    }
    __host__ __device__ Complex &operator[](const int &index)
    {
        return this->lattice_vec[index];
    }
    __host__ __device__ const Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1) const
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_s * this->lat_c0 * this->lat_c1 + index_c0 * this->lat_c1 + index_c1;
        return this->lattice_vec[index];
    }
    __host__ __device__ Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_s * this->lat_c0 * this->lat_c1 + index_c0 * this->lat_c1 + index_c1;
        return this->lattice_vec[index];
    }
    __host__ __device__ LatticeGauge operator+(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] + other[i];
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator-(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] - other[i];
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator-() const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = -this->lattice_vec[i];
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator*(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] * other[i];
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator/(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] / other[i];
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator+(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] + other;
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator-(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] - other;
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator*(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] * other;
        }
        return result;
    }
    __host__ __device__ LatticeGauge operator/(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] / other;
        }
        return result;
    }
    __host__ __device__ bool operator==(const LatticeGauge &other) const
    {
        if (this->size != other.size)
        {
            return false;
        }
        for (int i = 0; i < this->size; ++i)
        {
            if (lattice_vec[i] != other[i])
            {
                return false;
            }
        }
        return true;
    }
    __host__ __device__ bool operator!=(const LatticeGauge &other) const
    {
        return !(*this == other);
    }
    __host__ __device__ void print(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
    {
    }
    __host__ __device__ void print()
    {
        for (int x = 0; x < lat_x; x++)
        {
            for (int y = 0; y < lat_y; y++)
            {
                for (int z = 0; z < lat_z; z++)
                {
                    for (int t = 0; t < lat_t; t++)
                    {
                        for (int s = 0; s < lat_s; s++)
                        {
                            for (int c0 = 0; c0 < lat_c0; c0++)
                            {
                                for (int c1 = 0; c1 < lat_c1; c1++)
                                {
                                    print(x, y, z, t, s, c0, c1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __host__ __device__ double norm_2()
    {
        double result = 0;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].real * this->lattice_vec[i].real + this->lattice_vec[i].imag * this->lattice_vec[i].imag;
        }
        return result;
    }
    __host__ double norm_2X()
    {
        double local_result = 0;
        double global_result = 0;
        local_result = norm_2();
        MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_result;
    }
    __host__ __device__ Complex dot(const LatticeGauge &other)
    {
        Complex result;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].conj() * other[i];
        }
        return result;
    }
    __host__ Complex dotX(const LatticeGauge &other)
    {
        Complex local_result;
        Complex global_result;
        local_result = dot(other);
        MPI_Allreduce(&local_result, &global_result, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return global_result;
    }
};
__global__ void dslash(LatticeGauge &U, LatticeFermi &src, LatticeFermi &dest)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int t = threadIdx.x;
    const Complex i(0.0, 1.0);
    Complex tmp0[3];
    Complex tmp1[3];
    Complex g0[2];
    Complex g1[2];
    int s0[2];
    int s1[2];
    int d;
    double coef[2];
    coef[0] = 0;
    coef[1] = 1;
    Complex flag0;
    Complex flag1;
    // mass term and others
    for (int s = 0; s < U.lat_s; s++)
    {
        for (int c = 0; c < U.lat_c0; c++)
        {
            dest(x, y, z, t, s, c) += src(x, y, z, t, s, c) * coef[0];
        }
    }
    // backward x
    int b_x = (x + U.lat_x - 1) % U.lat_x;
    d = 0;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 3;
    g0[1] = i;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 2;
    g1[1] = i;
    flag0 = -i;
    flag1 = -i;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(b_x, y, z, t, s0[0], c1) * g0[0] + src(b_x, y, z, t, s0[1], c1) * g0[1]) * U(b_x, y, z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
            tmp1[c0] += (src(b_x, y, z, t, s1[0], c1) * g1[0] + src(b_x, y, z, t, s1[1], c1) * g1[1]) * U(b_x, y, z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp1[c0] * flag1;
        dest(x, y, z, t, 3, c0) += tmp0[c0] * flag0;
    }
    // forward x
    int f_x = (x + 1) % U.lat_x;
    d = 0;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 3;
    g0[1] = -i;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 2;
    g1[1] = -i;
    flag0 = i;
    flag1 = i;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(f_x, y, z, t, s0[0], c1) * g0[0] + src(f_x, y, z, t, s0[1], c1) * g0[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
            tmp1[c0] += (src(f_x, y, z, t, s1[0], c1) * g1[0] + src(f_x, y, z, t, s1[1], c1) * g1[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp1[c0] * flag1;
        dest(x, y, z, t, 3, c0) += tmp0[c0] * flag0;
    }
    // backward y
    int b_y = (y + U.lat_y - 1) % U.lat_y;
    d = 1;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 3;
    g0[1] = -1;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 2;
    g1[1] = 1;
    flag0 = -1;
    flag1 = 1;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, b_y, z, t, s0[0], c1) * g0[0] + src(x, b_y, z, t, s0[1], c1) * g0[1]) * U(x, b_y, z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
            tmp1[c0] += (src(x, b_y, z, t, s1[0], c1) * g1[0] + src(x, b_y, z, t, s1[1], c1) * g1[1]) * U(x, b_y, z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp1[c0] * flag1;
        dest(x, y, z, t, 3, c0) += tmp0[c0] * flag0;
    }
    // forward y
    int f_y = (y + 1) % U.lat_y;
    d = 1;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 3;
    g0[1] = 1;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 2;
    g1[1] = -1;
    flag0 = 1;
    flag1 = -1;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, f_y, z, t, s0[0], c1) * g0[0] + src(x, f_y, z, t, s0[1], c1) * g0[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
            tmp1[c0] += (src(x, f_y, z, t, s1[0], c1) * g1[0] + src(x, f_y, z, t, s1[1], c1) * g1[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp1[c0] * flag1;
        dest(x, y, z, t, 3, c0) += tmp0[c0] * flag0;
    }
    // backward z
    int b_z = (z + U.lat_z - 1) % U.lat_z;
    d = 2;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 2;
    g0[1] = i;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 3;
    g1[1] = -i;
    flag0 = -i;
    flag1 = i;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, y, b_z, t, s0[0], c1) * g0[0] + src(x, y, b_z, t, s0[1], c1) * g0[1]) * U(x, y, b_z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
            tmp1[c0] += (src(x, y, b_z, t, s1[0], c1) * g1[0] + src(x, y, b_z, t, s1[1], c1) * g1[1]) * U(x, y, b_z, t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp0[c0] * flag0;
        dest(x, y, z, t, 3, c0) += tmp1[c0] * flag1;
    }
    // forward z
    int f_z = (z + 1) % U.lat_z;
    d = 2;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 2;
    g0[1] = -i;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 3;
    g1[1] = i;
    flag0 = i;
    flag1 = -i;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, y, f_z, t, s0[0], c1) * g0[0] + src(x, y, f_z, t, s0[1], c1) * g0[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
            tmp1[c0] += (src(x, y, f_z, t, s1[0], c1) * g1[0] + src(x, y, f_z, t, s1[1], c1) * g1[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp0[c0] * flag0;
        dest(x, y, z, t, 3, c0) += tmp1[c0] * flag1;
    }
    // backward t
    int b_t = (t + U.lat_t - 1) % U.lat_t;
    d = 3;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 2;
    g0[1] = 1;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 3;
    g1[1] = 1;
    flag0 = 1;
    flag1 = 1;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, y, z, b_t, s0[0], c1) * g0[0] + src(x, y, z, b_t, s0[1], c1) * g0[1]) * U(x, y, z, b_t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
            tmp1[c0] += (src(x, y, z, b_t, s1[0], c1) * g1[0] + src(x, y, z, b_t, s1[1], c1) * g1[1]) * U(x, y, z, b_t, d, c1, c0).conj() * coef[1]; // what the fuck ? Hermitian operator！
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp0[c0] * flag0;
        dest(x, y, z, t, 3, c0) += tmp1[c0] * flag1;
    }
    // forward t
    int f_t = (t + 1) % U.lat_t;
    d = 3;
    tmp0[0] = 0;
    tmp0[1] = 0;
    tmp0[2] = 0;
    tmp1[0] = 0;
    tmp1[1] = 0;
    tmp1[2] = 0;
    s0[0] = 0;
    g0[0] = 1;
    s0[1] = 2;
    g0[1] = -1;
    s1[0] = 1;
    g1[0] = 1;
    s1[1] = 3;
    g1[1] = -1;
    flag0 = -1;
    flag1 = -1;
    for (int c0 = 0; c0 < U.lat_c0; c0++)
    {
        for (int c1 = 0; c1 < U.lat_c1; c1++)
        {
            tmp0[c0] += (src(x, y, z, f_t, s0[0], c1) * g0[0] + src(x, y, z, f_t, s0[1], c1) * g0[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
            tmp1[c0] += (src(x, y, z, f_t, s1[0], c1) * g1[0] + src(x, y, z, f_t, s1[1], c1) * g1[1]) * U(x, y, z, t, d, c0, c1) * coef[1];
        }
        dest(x, y, z, t, 0, c0) += tmp0[c0];
        dest(x, y, z, t, 1, c0) += tmp1[c0];
        dest(x, y, z, t, 2, c0) += tmp0[c0] * flag0;
        dest(x, y, z, t, 3, c0) += tmp1[c0] * flag1;
    }
}

int main()
{
    int lat_x(16);
    int lat_y(16);
    int lat_z(16);
    int lat_t(32);
    int lat_s(4);
    int lat_c(3);
    int size = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c;
    LatticeGauge *_U;
    LatticeFermi *_src;
    LatticeFermi *_dest;
    cudaMallocManaged(&_U, sizeof(LatticeGauge));
    cudaMallocManaged(&_src, sizeof(LatticeFermi));
    cudaMallocManaged(&_dest, sizeof(LatticeFermi));
    cudaMallocManaged(&_U->lattice_vec, lat_c * size * 2 * sizeof(double));
    cudaMallocManaged(&_src->lattice_vec, size * 2 * sizeof(double));
    cudaMallocManaged(&_dest->lattice_vec, size * 2 * sizeof(double));
    LatticeGauge &U = *_U;
    LatticeFermi &src = *_src;
    LatticeFermi &dest = *_dest;
    U.lat_x = lat_x;
    U.lat_y = lat_y;
    U.lat_z = lat_z;
    U.lat_t = lat_t;
    U.lat_s = lat_s;
    U.lat_c0 = lat_c;
    U.lat_c1 = lat_c;
    U.size = U.lat_x * U.lat_y * U.lat_z * U.lat_t * U.lat_s * U.lat_c0 * U.lat_c1;
    src.lat_x = lat_x;
    src.lat_y = lat_y;
    src.lat_z = lat_z;
    src.lat_t = lat_t;
    src.lat_s = lat_s;
    src.lat_c = lat_c;
    src.size = src.lat_x * src.lat_y * src.lat_z * src.lat_t * src.lat_s * src.lat_c;
    dest.lat_x = lat_x;
    dest.lat_y = lat_y;
    dest.lat_z = lat_z;
    dest.lat_t = lat_t;
    dest.lat_s = lat_s;
    dest.lat_c = lat_c;
    dest.size = dest.lat_x * dest.lat_y * dest.lat_z * dest.lat_t * dest.lat_s * dest.lat_c;
    U.assign_random(666);
    src.assign_random(111);
    dest.assign_zero();
    dim3 gridSize(lat_x, lat_y, lat_z);
    dim3 blockSize(lat_t);
    std::cout << "src.norm_2():" << src.norm_2() << std::endl;
    std::cout << "dest.norm_2():" << dest.norm_2() << std::endl;
    clock_t start = clock();
    dslash<<<gridSize, blockSize>>>(U, src, dest);
    cudaDeviceSynchronize();
    clock_t end = clock();
    std::cout << "src.norm_2():" << src.norm_2() << std::endl;
    std::cout << "dest.norm_2():" << dest.norm_2() << std::endl;
    std::cout
        << "################"
        << "time cost:"
        << (double)(end - start) / CLOCKS_PER_SEC
        << "s"
        << std::endl;
    //// MPI_Finalize();
    return 0;
}