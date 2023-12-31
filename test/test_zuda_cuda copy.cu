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
    __host__ LatticeFermi block(const int &num_x, const int &num_y, const int &num_z, const int &num_t, const int &index_x, const int &index_y, const int &index_z, const int &index_t)
    {
        int block_x, block_y, block_z, block_t;
        block_x = this->lat_x / num_x;
        block_y = this->lat_y / num_y;
        block_z = this->lat_z / num_z;
        block_t = this->lat_t / num_t;
        int start_x = index_x * block_x;
        int start_y = index_y * block_y;
        int start_z = index_z * block_z;
        int start_t = index_t * block_t;
        LatticeFermi result(block_x, block_y, block_z, block_t, this->lat_s, this->lat_c);
        int global_x, global_y, global_z, global_t;
        for (int x = 0; x < block_x; x++)
        {
            global_x = start_x + x;
            for (int y = 0; y < block_y; y++)
            {
                global_y = start_y + y;
                for (int z = 0; z < block_z; z++)
                {
                    global_z = start_z + z;
                    for (int t = 0; t < block_t; t++)
                    {
                        global_t = start_t + t;
                        for (int s = 0; s < this->lat_s; s++)
                        {
                            for (int c = 0; c < this->lat_c; c++)
                            {
                                result(x, y, z, t, s, c) = (*this)(global_x, global_y, global_z, global_t, s, c);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
    __host__ LatticeFermi block(const int &num_x, const int &num_y, const int &num_z, const int &num_t)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return block(num_x, num_y, num_z, num_t,
                     rank / (num_y * num_z * num_t),
                     (rank / (num_z * num_t)) % num_y,
                     (rank / num_t) % num_z,
                     rank % num_t);
    }
    __host__ LatticeFermi reback(const int &num_x, const int &num_y, const int &num_z, const int &num_t)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int start_x = (rank / (num_y * num_z * num_t)) * this->lat_x;
        int start_y = ((rank / (num_z * num_t)) % num_y) * this->lat_y;
        int start_z = ((rank / num_t) % num_z) * this->lat_z;
        int start_t = (rank % num_t) * this->lat_t;
        LatticeFermi result(num_x * this->lat_x, num_y * this->lat_y, num_z * this->lat_z, num_t * this->lat_t, this->lat_s, this->lat_c);
        int global_x, global_y, global_z, global_t;
        for (int x = 0; x < this->lat_x; x++)
        {
            global_x = start_x + x;
            for (int y = 0; y < this->lat_y; y++)
            {
                global_y = start_y + y;
                for (int z = 0; z < this->lat_z; z++)
                {
                    global_z = start_z + z;
                    for (int t = 0; t < this->lat_t; t++)
                    {
                        global_t = start_t + t;
                        for (int s = 0; s < this->lat_s; s++)
                        {
                            for (int c = 0; c < this->lat_c; c++)
                            {
                                result(global_x, global_y, global_z, global_t, s, c) = (*this)(x, y, z, t, s, c);
                            }
                        }
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, result.lattice_vec, result.size * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
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
    __host__ LatticeGauge block(const int &num_x, const int &num_y, const int &num_z, const int &num_t, const int &index_x, const int &index_y, const int &index_z, const int &index_t)
    {
        int block_x, block_y, block_z, block_t;
        block_x = this->lat_x / num_x;
        block_y = this->lat_y / num_y;
        block_z = this->lat_z / num_z;
        block_t = this->lat_t / num_t;
        int start_x = index_x * block_x;
        int start_y = index_y * block_y;
        int start_z = index_z * block_z;
        int start_t = index_t * block_t;
        LatticeGauge result(block_x, block_y, block_z, block_t, lat_s, lat_c0);
        int global_x, global_y, global_z, global_t;
        for (int x = 0; x < block_x; x++)
        {
            global_x = start_x + x;
            for (int y = 0; y < block_y; y++)
            {
                global_y = start_y + y;
                for (int z = 0; z < block_z; z++)
                {
                    global_z = start_z + z;
                    for (int t = 0; t < block_t; t++)
                    {
                        global_t = start_t + t;
                        for (int s = 0; s < this->lat_s; s++)
                        {
                            for (int c0 = 0; c0 < this->lat_c0; c0++)
                            {
                                for (int c1 = 0; c1 < this->lat_c1; c1++)
                                {
                                    result(x, y, z, t, s, c0, c1) = (*this)(global_x, global_y, global_z, global_t, s, c0, c1);
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
    __host__ LatticeGauge block(const int &num_x, const int &num_y, const int &num_z, const int &num_t)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return block(num_x, num_y, num_z, num_t,
                     rank / (num_y * num_z * num_t),
                     (rank / (num_z * num_t)) % num_y,
                     (rank / num_t) % num_z,
                     rank % num_t);
    }
    __host__ LatticeGauge reback(const int &num_x, const int &num_y, const int &num_z, const int &num_t)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int start_x = (rank / (num_y * num_z * num_t)) * this->lat_x;
        int start_y = ((rank / (num_z * num_t)) % num_y) * this->lat_y;
        int start_z = ((rank / num_t) % num_z) * this->lat_z;
        int start_t = (rank % num_t) * this->lat_t;
        LatticeGauge result(num_x * this->lat_x, num_y * this->lat_y, num_z * this->lat_z, num_t * this->lat_t, this->lat_s, this->lat_c0);
        int global_x, global_y, global_z, global_t;
        for (int x = 0; x < this->lat_x; x++)
        {
            global_x = start_x + x;
            for (int y = 0; y < this->lat_y; y++)
            {
                global_y = start_y + y;
                for (int z = 0; z < this->lat_z; z++)
                {
                    global_z = start_z + z;
                    for (int t = 0; t < this->lat_t; t++)
                    {
                        global_t = start_t + t;
                        for (int s = 0; s < this->lat_s; s++)
                        {
                            for (int c0 = 0; c0 < this->lat_c0; c0++)
                            {
                                for (int c1 = 0; c1 < this->lat_c1; c1++)
                                {
                                    result(global_x, global_y, global_z, global_t, s, c0, c1) = (*this)(x, y, z, t, s, c0, c1);
                                }
                            }
                        }
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, result.lattice_vec, result.size * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
    }
};
__host__ __device__ void dslash_test(LatticeGauge &U, LatticeFermi &src, LatticeFermi &dest)
{
    for (int i = 0; i < dest.size; i++)
    {
        dest.lattice_vec[i] = src.lattice_vec[i] * 0.5;
    }

    dest.lattice_vec[0] *= 2;
}
__global__ void dslash(LatticeGauge &U, LatticeFermi &src, LatticeFermi &dest, const int *inthread_num, const int *inblock_num, const int *ingrid_num, const bool test)
{
    if (test)
    {
        dslash_test(U, src, dest);
        return;
    }
    int thread_index = threadIdx.x;
    int grid_index = blockIdx.x;
    int thread_size = blockDim.x;
    int grid_size = gridDim.x;

    dest.assign_zero();
    dest.assign_unit(); // test
    const Complex i(0.0, 1.0);
    Complex tmp0[3];
    Complex tmp1[3];
    Complex g0[2];
    Complex g1[2];
    int s0[2];
    int s1[2];
    int d;
    // const double a = 1.0;
    // const double kappa = 1.0;
    // const double m = -3.5;
    double coef[2];
    coef[0] = 0;
    coef[1] = 1;
    Complex flag0;
    Complex flag1;
    /*
    for (int x = 0; x < U.lat_x; x++)
    {
        for (int y = 0; y < U.lat_y; y++)
        {
            for (int z = 0; z < U.lat_z; z++)
            {
                for (int t = 0; t < U.lat_t; t++)
                {
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
            }
        }
    }
    */
}

//__host__ void cg(LatticeGauge &U, LatticeFermi &b, LatticeFermi &x, const int &num_x, const int &num_y, const int &num_z, const int &num_t, const int &MAX_ITER, const double &TOL, const double &test)
//{
//    Complex rho_prev(1.0, 0.0), rho(0.0, 0.0), alpha(1.0, 0.0), omega(1.0, 0.0), beta(0.0, 0.0);
//    double r_norm2 = 0;
//    LatticeFermi
//        r(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
//        r_tilde(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
//        p(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
//        v(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
//        s(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
//        t(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c);
//    // x.rand(); // initial guess
//    // // ComplexVector r = b - A * x;
//    x.assign_random(666);
//    int size=num_x*num_y*num_z*num_t;
//    dim3 blockSize_(256);
//    dim3 gridSize_(size);
//    dslash<< < gridSize_, blockSize_ >> >(U, x, r,test);;
//    r = b - r;
//    r_tilde = r;
//    // r.print();
//    // if x=0;r_tilde = r0 = b;
//    // x.assign_zero();
//    // r = b;
//    // r_tilde = r;
//    for (int i = 0; i < MAX_ITER; i++)
//    {
//        rho = r_tilde.dotX(r);
//        beta = (rho / rho_prev) * (alpha / omega);
//        p = r + (p - v * omega) * beta;
//        // v = A * p;
//        dslash<< < gridSize_, blockSize_ >> >(U, p, v,test);;
//        alpha = rho / r_tilde.dotX(v);
//        s = r - v * alpha;
//        // t = A * s;
//        dslash<< < gridSize_, blockSize_ >> >(U, s, t,test);;
//        omega = t.dotX(s) / t.dotX(t);
//        x = x + p * alpha + s * omega;
//        r = s - t * omega;
//        r_norm2 = r.norm_2X();
//        std::cout << "##loop "
//                  << i
//                  << "##Residual:"
//                  << r_norm2
//                  << std::endl;
//        // break;
//        if (r_norm2 < TOL || i == MAX_ITER - 1)
//        {
//            break;
//        }
//        rho_prev = rho;
//    }
//}

// void ext_dslash(const double *U_real, const double *U_imag, const double *src_real, const double *src_imag, const double *dest_real, const double *dest_imag, const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c, const int &num_x, const int &num_y, const int &num_z, const int &num_t, const bool &test)
//{
//    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    LatticeFermi src(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    LatticeFermi dest(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    for (int i = 0; i < U.size; i++)
//    {
//        U[i].real = U_real[i];
//        U[i].imag = U_imag[i];
//    }
//    for (int i = 0; i < src.size; i++)
//    {
//        src[i].real = src_real[i];
//        src[i].imag = src_imag[i];
//        dest[i].real = dest_real[i];
//        dest[i].imag = dest_imag[i];
//    }
//    dslash(U, src, dest,test);;
//}
// void ext_cg(const double *U_real, const double *U_imag, const double *b_real, const double *b_imag, const double *x_real, const double *x_imag, const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c, const int &num_x, const int &num_y, const int &num_z, const int &num_t, const int MAX_ITER, const double TOL, const bool &test)
//{
//    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    LatticeFermi b(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    LatticeFermi x(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
//    for (int i = 0; i < U.size; i++)
//    {
//        U[i].real = U_real[i];
//        U[i].imag = U_imag[i];
//    }
//    for (int i = 0; i < b.size; i++)
//    {
//        b[i].real = b_real[i];
//        b[i].imag = b_imag[i];
//        x[i].real = x_real[i];
//        x[i].imag = x_imag[i];
//    }
//    cg(U, b, x, num_x, num_y, num_z, num_t, MAX_ITER, TOL, test);
//}
__host__ void lattice_block(const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &maxThreadsPerBlock, const int &max_block_num, int &block_num, int *inthread_num, int *inblock_num, int *ingrid_num)
{
    int tmp[5];
    int inblock_log2(0);
    int lat_log2[5];
    tmp[0] = lat_x;
    tmp[1] = lat_y;
    tmp[2] = lat_z;
    tmp[3] = lat_t;
    tmp[4] = maxThreadsPerBlock;
    lat_log2[4] = 0;
    inblock_log2 = 0;
    do
    {
        if (tmp[4] % 2)
        {
            break;
        }
        else
        {
            lat_log2[4] += 1;
            tmp[4] /= 2;
        }
    } while (true);
    for (int d = 0; d < 4; d++)
    {
        inthread_num[d] = 1;
        inblock_num[d] = 1;
        ingrid_num[d] = 1;
        lat_log2[d] = 0;
        do
        {
            std::cout << "tmp[" << d << "]:" << tmp[d] << std::endl;
            if (tmp[d] % 2)
            {
                inthread_num[d] *= tmp[d];
                std::cout << " inthread_num[" << d << "]:" << inthread_num[d] << std::endl;
                break;
            }
            else
            {
                tmp[d] /= 2;
                lat_log2[d] += 1;
                std::cout << "lat_log2[" << d << "]:" << lat_log2[d] << std::endl;
                inblock_log2 += 1;
                std::cout << "inblock_log2:" << inblock_log2 << std::endl;
                if (inblock_log2 == lat_log2[4] + 1)
                {
                    inblock_log2 -= 1;
                    block_num *= 2;
                    if (block_num > max_block_num)
                    {
                        block_num /= 2;
                        ingrid_num[d] /= 2;
                        inthread_num[d] *= 2;
                        std::cout << " inthread_num[" << d << "]:" << inthread_num[d] << std::endl;
                    }
                    std::cout << " block_num:" << block_num << std::endl;
                    ingrid_num[d] *= 2;
                    std::cout << " ingrid_num[" << d << "]:" << ingrid_num[d] << std::endl;
                    continue;
                }
                inblock_num[d] *= 2;
                std::cout << " inblock_num[" << d << "]:" << inblock_num[d] << std::endl;
            }
        } while (true);
    }
}

// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
    int width;
    int height;
    float *elements;
};
// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
    A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < A->width; ++i)
    {
        Cvalue += getElement(A, row, i) * getElement(B, i, col);
    }
    setElement(C, row, col, Cvalue);
}

int main()
{
    // MPI_Init(NULL, NULL);
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "Use GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "Number of SMs:" << devProp.multiProcessorCount << std::endl;
    std::cout << "Shared memory size of each thread block:" << devProp.sharedMemPerBlock / 1024.0 << "KB" << std::endl;
    std::cout << "The maximum number of threads per thread block:" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "The maximum number of threads per EM:" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Maximum number of warps per EM:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    int width = 1 << 10;
    int height = 1 << 10;
    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void **)&A, sizeof(Matrix));
    cudaMallocManaged((void **)&B, sizeof(Matrix));
    cudaMallocManaged((void **)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void **)&A->elements, nBytes);
    cudaMallocManaged((void **)&B->elements, nBytes);
    cudaMallocManaged((void **)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = 1.0;
        B->elements[i] = 2.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    matMulKernel<<<gridSize, blockSize>>>(A, B, C);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; ++i)
        maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    std::cout << "最大误差: " << maxError << std::endl;

    int lat_x(16);
    int lat_y(16);
    int lat_z(16);
    int lat_t(16);
    int lat_s(4);
    int lat_c(3);
    int size = lat_x * lat_y * lat_z * lat_t * lat_s * lat_c;
    bool test(false);
    // int MAX_ITER(1e6);
    // double TOL(1e-6);
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
    cudaDeviceSynchronize();
    int inthread_num[4];
    int inblock_num[4];
    int ingrid_num[4];
    int block_num(1);
    int max_block_num(32 * 64);
    lattice_block(lat_x, lat_y, lat_z, lat_t, devProp.maxThreadsPerBlock, max_block_num, block_num, inthread_num, inblock_num, ingrid_num);
    dim3 gridSize_(block_num);
    dim3 blockSize_(devProp.maxThreadsPerBlock);
    std::cout << " block_num:" << block_num << std::endl;
    for (int d = 0; d < 4; d++)
    {
        std::cout << "################" << std::endl;
        std::cout << " inthread_num[" << d << "]:" << inthread_num[d] << std::endl;
        std::cout << " inblock_num[" << d << "]:" << inblock_num[d] << std::endl;
        std::cout << " ingrid_num[" << d << "]:" << ingrid_num[d] << std::endl;
    }
    std::cout << "src.norm_2():" << src.norm_2() << std::endl;
    std::cout << "dest.norm_2():" << dest.norm_2() << std::endl;
    clock_t start = clock();
    dslash<<<gridSize_, blockSize_>>>(U, src, dest, inthread_num, inblock_num, ingrid_num,test);
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