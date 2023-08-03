#include <iostream>
#include <random>
#include <time.h>
const int lat_x = 16;
const int lat_y = 16;
const int lat_z = 16;
const int lat_t = 16;
const int lat_s = 4;
const int lat_c = 3;
const int lat_c0 = 3;
const int lat_c1 = 3;
class Complex
{
public:
    double data[2];
    Complex(double real = 0.0, double imag = 0.0)
    {
        this->data[0] = real;
        this->data[1] = imag;
    }
    Complex &operator=(const Complex &other)
    {
        if (this != &other)
        {
            this->data[0] = other.data[0];
            this->data[1] = other.data[1];
        }
        return *this;
    }
    Complex operator+(const Complex &other) const
    {
        return Complex(this->data[0] + other.data[0], this->data[1] + other.data[1]);
    }
    Complex operator-(const Complex &other) const
    {
        return Complex(this->data[0] - other.data[0], this->data[1] - other.data[1]);
    }
    Complex operator*(const Complex &other) const
    {
        return Complex(this->data[0] * other.data[0] - this->data[1] * other.data[1],
                       this->data[0] * other.data[1] + this->data[1] * other.data[0]);
    }
    Complex operator*(const double &other) const
    {
        return Complex(this->data[0] * other, this->data[1] * other);
    }
    Complex operator/(const Complex &other) const
    {
        double denom = other.data[0] * other.data[0] + other.data[1] * other.data[1];
        return Complex((this->data[0] * other.data[0] + this->data[1] * other.data[1]) / denom,
                       (this->data[1] * other.data[0] - this->data[0] * other.data[1]) / denom);
    }
    Complex operator/(const double &other) const
    {
        return Complex(this->data[0] / other, this->data[1] / other);
    }
    Complex operator-() const
    {
        return Complex(-this->data[0], -this->data[1]);
    }
    Complex &operator+=(const Complex &other)
    {
        this->data[0] += other.data[0];
        this->data[1] += other.data[1];
        return *this;
    }
    Complex &operator-=(const Complex &other)
    {
        this->data[0] -= other.data[0];
        this->data[1] -= other.data[1];
        return *this;
    }
    Complex &operator*=(const Complex &other)
    {
        this->data[0] = this->data[0] * other.data[0] - this->data[1] * other.data[1];
        this->data[1] = this->data[0] * other.data[1] + this->data[1] * other.data[0];
        return *this;
    }
    Complex &operator*=(const double &scalar)
    {
        this->data[0] *= scalar;
        this->data[1] *= scalar;
        return *this;
    }
    Complex &operator/=(const Complex &other)
    {
        double denom = other.data[0] * other.data[0] + other.data[1] * other.data[1];
        this->data[0] = (data[0] * other.data[0] + data[1] * other.data[1]) / denom;
        this->data[1] = (data[1] * other.data[0] - data[0] * other.data[1]) / denom;
        return *this;
    }
    Complex &operator/=(const double &other)
    {
        this->data[0] /= other;
        this->data[1] /= other;
        return *this;
    }
    bool operator==(const Complex &other) const
    {
        return (this->data[0] == other.data[0] && this->data[1] == other.data[1]);
    }
    bool operator!=(const Complex &other) const
    {
        return !(*this == other);
    }
    friend std::ostream &operator<<(std::ostream &os, const Complex &c)
    {
        if (c.data[1] >= 0.0)
        {
            os << c.data[0] << " + " << c.data[1] << "i";
        }
        else
        {
            os << c.data[0] << " - " << std::abs(c.data[1]) << "i";
        }
        return os;
    }
    Complex conj()
    {
        return Complex(this->data[0], -this->data[1]);
    }
};
const Complex i(0.0, 1.0);
const Complex zero(0.0, 0.0);
class LatticeFermi
{
public:
    int lat_x, lat_y, lat_z, lat_t, lat_s, lat_c;
    int size;
    Complex *lattice_vec;
    LatticeFermi(const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c)
        : lat_x(lat_x), lat_y(lat_y), lat_z(lat_z), lat_t(lat_t), lat_s(lat_s), lat_c(lat_c), size(lat_x * lat_y * lat_z * lat_t * lat_s * lat_c)
    {
        this->lattice_vec = new Complex[size];
    }
    ~LatticeFermi()
    {
        if (this->lattice_vec != nullptr)
        {
            this->lattice_vec = nullptr;
            delete[] this->lattice_vec;
        }
    }
    LatticeFermi &operator=(const LatticeFermi &other)
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
    void assign_zero()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = 0;
            this->lattice_vec[i].data[1] = 0;
        }
    }
    void assign_unit()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = 1;
            this->lattice_vec[i].data[1] = 0;
        }
    }
    void assign_random(unsigned seed = 32767)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = u(e);
            this->lattice_vec[i].data[1] = u(e);
        }
    }
    void info()
    {
        std::cout << "lat_x:" << this->lat_x << std::endl;
        std::cout << "lat_y:" << this->lat_y << std::endl;
        std::cout << "lat_z:" << this->lat_z << std::endl;
        std::cout << "lat_t:" << this->lat_t << std::endl;
        std::cout << "lat_s:" << this->lat_s << std::endl;
        std::cout << "lat_c:" << this->lat_c << std::endl;
        std::cout << "size:" << this->size << std::endl;
    }
    const Complex &operator[](const int &index) const
    {
        return this->lattice_vec[index];
    }
    Complex &operator[](const int &index)
    {
        return this->lattice_vec[index];
    }
    const Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c) const
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_z * this->lat_t * this->lat_s * this->lat_c + index_t * this->lat_s * this->lat_c + index_s * this->lat_c + index_c;
        return this->lattice_vec[index];
    }
    Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c + index_z * this->lat_t * this->lat_s * this->lat_c + index_t * this->lat_s * this->lat_c + index_s * this->lat_c + index_c;
        return this->lattice_vec[index];
    }
    LatticeFermi operator+(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] + other[i];
        }
        return result;
    }
    LatticeFermi operator-(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] - other[i];
        }
        return result;
    }
    LatticeFermi operator-() const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = -this->lattice_vec[i];
        }
        return result;
    }
    LatticeFermi operator*(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] * other[i];
        }
        return result;
    }
    LatticeFermi operator/(const LatticeFermi &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] / other[i];
        }
        return result;
    }
    LatticeFermi operator+(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] + other;
        }
        return result;
    }
    LatticeFermi operator-(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] - other;
        }
        return result;
    }
    LatticeFermi operator*(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] * other;
        }
        return result;
    }
    LatticeFermi operator/(const Complex &other) const
    {
        LatticeFermi result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] / other;
        }
        return result;
    }
    bool operator==(const LatticeFermi &other) const
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
    bool operator!=(const LatticeFermi &other) const
    {
        return !(*this == other);
    }
    void print(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
    {
        std::cout << "lattice_vec[" << index_x << "][" << index_y << "][" << index_z << "][" << index_t << "][" << index_s << "][" << index_c << "] = " << (*this)(index_x, index_y, index_z, index_t, index_s, index_c) << std::endl;
    }
    void print()
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
    double norm_2()
    {
        double result = 0;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].data[0] * this->lattice_vec[i].data[0] + this->lattice_vec[i].data[1] * this->lattice_vec[i].data[1];
        }
        return result;
    }
    Complex dot(const LatticeFermi &other)
    {
        Complex result;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].conj() * other[i];
        }
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
    LatticeGauge(const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c)
        : lat_x(lat_x), lat_y(lat_y), lat_z(lat_z), lat_t(lat_t), lat_s(lat_s), lat_c0(lat_c), lat_c1(lat_c), size(lat_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1)
    {
        this->lattice_vec = new Complex[size];
    }
    ~LatticeGauge()
    {
        if (this->lattice_vec != nullptr)
        {
            this->lattice_vec = nullptr;
            delete[] this->lattice_vec;
        }
    }
    LatticeGauge &operator=(const LatticeGauge &other)
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
    void assign_zero()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = 0;
            this->lattice_vec[i].data[1] = 0;
        }
    }
    void assign_unit()
    {
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = 1;
            this->lattice_vec[i].data[1] = 0;
        }
    }
    void assign_random(unsigned seed = 32767)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < this->size; i++)
        {
            this->lattice_vec[i].data[0] = u(e);
            this->lattice_vec[i].data[1] = u(e);
        }
    }
    void info()
    {
        std::cout << "lat_x:" << this->lat_x << std::endl;
        std::cout << "lat_y:" << this->lat_y << std::endl;
        std::cout << "lat_z:" << this->lat_z << std::endl;
        std::cout << "lat_t:" << this->lat_t << std::endl;
        std::cout << "lat_s:" << this->lat_s << std::endl;
        std::cout << "lat_c0:" << this->lat_c0 << std::endl;
        std::cout << "lat_c1:" << this->lat_c1 << std::endl;
        std::cout << "size:" << this->size << std::endl;
    }
    const Complex &operator[](const int &index) const
    {
        return this->lattice_vec[index];
    }
    Complex &operator[](const int &index)
    {
        return this->lattice_vec[index];
    }
    const Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1) const
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_s * this->lat_c0 * this->lat_c1 + index_c0 * this->lat_c1 + index_c1;
        return this->lattice_vec[index];
    }
    Complex &operator()(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
    {
        int index = index_x * this->lat_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_y * this->lat_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_z * this->lat_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_t * this->lat_s * this->lat_c0 * this->lat_c1 + index_s * this->lat_c0 * this->lat_c1 + index_c0 * this->lat_c1 + index_c1;
        return this->lattice_vec[index];
    }
    LatticeGauge operator+(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] + other[i];
        }
        return result;
    }
    LatticeGauge operator-(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] - other[i];
        }
        return result;
    }
    LatticeGauge operator-() const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = -this->lattice_vec[i];
        }
        return result;
    }
    LatticeGauge operator*(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] * other[i];
        }
        return result;
    }
    LatticeGauge operator/(const LatticeGauge &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result[i] = this->lattice_vec[i] / other[i];
        }
        return result;
    }
    LatticeGauge operator+(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] + other;
        }
        return result;
    }
    LatticeGauge operator-(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] - other;
        }
        return result;
    }
    LatticeGauge operator*(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] * other;
        }
        return result;
    }
    LatticeGauge operator/(const Complex &other) const
    {
        LatticeGauge result(this->lat_x, this->lat_y, this->lat_z, this->lat_t, this->lat_s, this->lat_c0);
        for (int i = 0; i < this->size; ++i)
        {
            result.lattice_vec[i] = this->lattice_vec[i] / other;
        }
        return result;
    }
    bool operator==(const LatticeGauge &other) const
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
    bool operator!=(const LatticeGauge &other) const
    {
        return !(*this == other);
    }
    void print(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
    {
        std::cout << "lattice_vec[" << index_x << "][" << index_y << "][" << index_z << "][" << index_t << "][" << index_s << "][" << index_c0 << "][" << index_c1 << "] = " << (*this)(index_x, index_y, index_z, index_t, index_s, index_c0, index_c1) << std::endl;
    }
    void print()
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
    double norm_2()
    {
        double result = 0;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].data[0] * this->lattice_vec[i].data[0] + this->lattice_vec[i].data[1] * this->lattice_vec[i].data[1];
        }
        return result;
    }
    Complex dot(const LatticeGauge &other)
    {
        Complex result;
        for (int i = 0; i < this->size; i++)
        {
            result = result + this->lattice_vec[i].conj() * other[i];
        }
        return result;
    }
};
int index_guage(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c0, const int &index_c1)
{
    return index_x * lat_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1 + index_y * lat_z * lat_t * lat_s * lat_c0 * lat_c1 + index_z * lat_t * lat_s * lat_c0 * lat_c1 + index_t * lat_s * lat_c0 * lat_c1 + index_s * lat_c0 * lat_c1 + index_c0 * lat_c1 + index_c1;
}
int index_fermi(const int &index_x, const int &index_y, const int &index_z, const int &index_t, const int &index_s, const int &index_c)
{
    return index_x * lat_y * lat_z * lat_t * lat_s * lat_c + index_y * lat_z * lat_t * lat_s * lat_c + index_z * lat_t * lat_s * lat_c + index_t * lat_s * lat_c + index_s * lat_c + index_c;
}
void dslash_test(LatticeGauge &U, LatticeFermi &src, LatticeFermi &dest)
{
    for (int i = 0; i < dest.size; i++)
    {
        dest.lattice_vec[i] = src.lattice_vec[i] * 0.5;
    }
    dest.lattice_vec[0] *= 2;
}
void dslash(LatticeGauge &U, LatticeFermi &src, LatticeFermi &dest, const bool test)
{
    std::cout << "######U.norm_2():" << U.norm_2() << std::endl;
    std::cout << "######src.norm_2():" << src.norm_2() << std::endl;
    if (test)
    {
        dslash_test(U, src, dest);
        return;
    }
    dest.assign_zero();
    int tmp;
    Complex tmp0;
    Complex tmp1;
    clock_t start = clock();
    for (int x = 0; x < U.lat_x; x++)
    {
        for (int y = 0; y < U.lat_y; y++)
        {
            for (int z = 0; z < U.lat_z; z++)
            {
                for (int t = 0; t < U.lat_t; t++)
                {
                    // backward x
                    if (x == 0)
                    {
                        tmp = lat_x - 1;
                    }
                    else
                    {
                        tmp = x - 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] + src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
                            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] + src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(tmp, y, z, t, 0, c1, c0)].conj();
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1 * i;
                        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0 * i;
                    }
                    // forward x
                    if (x == lat_x - 1)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = x + 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(tmp, y, z, t, 0, c1)] - src[index_fermi(tmp, y, z, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
                            tmp1 += (src[index_fermi(tmp, y, z, t, 1, c1)] - src[index_fermi(tmp, y, z, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 0, c0, c1)];
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1 * i;
                        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0 * i;
                    }
                    // backward y
                    if (y == 0)
                    {
                        tmp = lat_y - 1;
                    }
                    else
                    {
                        tmp = y - 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] - src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
                            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] + src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, tmp, z, t, 1, c1, c0)].conj();
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp0;
                    }
                    // forward y
                    if (y == lat_y - 1)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = y + 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, tmp, z, t, 0, c1)] + src[index_fermi(x, tmp, z, t, 3, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
                            tmp1 += (src[index_fermi(x, tmp, z, t, 1, c1)] - src[index_fermi(x, tmp, z, t, 2, c1)]) * U[index_guage(x, y, z, t, 1, c0, c1)];
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp1;
                        dest[index_fermi(x, y, z, t, 3, c0)] += tmp0;
                    }
                    // backward z
                    if (z == 0)
                    {
                        tmp = lat_z - 1;
                    }
                    else
                    {
                        tmp = z - 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] + src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
                            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] - src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, tmp, t, 2, c1, c0)].conj();
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0 * i;
                        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1 * i;
                    }
                    // forward z
                    if (z == lat_z - 1)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = z + 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, y, tmp, t, 0, c1)] - src[index_fermi(x, y, tmp, t, 2, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
                            tmp1 += (src[index_fermi(x, y, tmp, t, 1, c1)] + src[index_fermi(x, y, tmp, t, 3, c1)] * i) * U[index_guage(x, y, z, t, 2, c0, c1)];
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0 * i;
                        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1 * i;
                    }
                    // backward t
                    if (t == 0)
                    {
                        tmp = lat_t - 1;
                    }
                    else
                    {
                        tmp = t - 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] + src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
                            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] + src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, tmp, 3, c1, c0)].conj();
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 3, c0)] += tmp1;
                    }
                    // forward t
                    if (t == lat_t - 1)
                    {
                        tmp = 0;
                    }
                    else
                    {
                        tmp = t + 1;
                    }
                    for (int c0 = 0; c0 < lat_c0; c0++)
                    {
                        tmp0 = zero;
                        tmp1 = zero;
                        for (int c1 = 0; c1 < lat_c1; c1++)
                        {
                            tmp0 += (src[index_fermi(x, y, z, tmp, 0, c1)] - src[index_fermi(x, y, z, tmp, 2, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
                            tmp1 += (src[index_fermi(x, y, z, tmp, 1, c1)] - src[index_fermi(x, y, z, tmp, 3, c1)]) * U[index_guage(x, y, z, t, 3, c0, c1)];
                        }
                        dest[index_fermi(x, y, z, t, 0, c0)] += tmp0;
                        dest[index_fermi(x, y, z, t, 1, c0)] += tmp1;
                        dest[index_fermi(x, y, z, t, 2, c0)] -= tmp0;
                        dest[index_fermi(x, y, z, t, 3, c0)] -= tmp1;
                    }
                }
            }
        }
    }
    clock_t end = clock();
    std::cout << "######dest.norm_2():" << dest.norm_2() << std::endl;
    std::cout << "######time cost:" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
}
void cg(LatticeGauge &U, const LatticeFermi &b, LatticeFermi &x, const int &MAX_ITER, const double &TOL, const double &test)
{
    Complex rho_prev(1.0, 0.0), rho(0.0, 0.0), alpha(1.0, 0.0), omega(1.0, 0.0), beta(0.0, 0.0);
    double r_norm2 = 0;
    LatticeFermi
        r(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        r_tilde(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        p(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        v(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        s(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c),
        t(b.lat_x, b.lat_y, b.lat_z, b.lat_t, b.lat_s, b.lat_c);
    // x.rand(); // initial guess
    // // ComplexVector r = b - A * x;
    x.assign_random(666);
    dslash(U, x, r, test);
    r = b - r;
    r_tilde = r;
    // r.print();
    // if x=0;r_tilde = r0 = b;
    // x.assign_zero();
    // r = b;
    // r_tilde = r;
    for (int i = 0; i < MAX_ITER; i++)
    {
        rho = r_tilde.dot(r);
        std::cout << "######rho:" << rho << " ######" << std::endl;
        beta = (rho / rho_prev) * (alpha / omega);
        std::cout << "######beta:" << beta << " ######" << std::endl;
        p = r + (p - v * omega) * beta;
        std::cout << "######p.norm_2():" << p.norm_2() << std::endl;
        // v = A * p;
        dslash(U, p, v, test);
        std::cout << "######v.norm_2():" << v.norm_2() << std::endl;
        alpha = rho / r_tilde.dot(v);
        std::cout << "######alpha:" << alpha << " ######" << std::endl;
        s = r - v * alpha;
        std::cout << "######s.norm_2():" << s.norm_2() << std::endl;
        // t = A * s;
        dslash(U, s, t, test);
        std::cout << "######t.norm_2():" << t.norm_2() << std::endl;
        omega = t.dot(s) / t.dot(t);
        std::cout << "######omega:" << omega << " ######" << std::endl;
        x = x + p * alpha + s * omega;
        std::cout << "######x.norm_2():" << x.norm_2() << std::endl;
        r = s - t * omega;
        r_norm2 = r.norm_2();
        std::cout << "######r.norm_2():" << r_norm2 << std::endl;
        std::cout << "##loop "
                  << i
                  << "##Residual:"
                  << r_norm2
                  << std::endl;
        // break;
        if (r_norm2 < TOL || i == MAX_ITER - 1)
        {
            x.print();
            break;
        }
        rho_prev = rho;
    }
}
void ext_dslash(const double *U_real, const double *U_imag, const double *src_real, const double *src_imag, double *dest_real, double *dest_imag, const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c, const bool &test)
{
    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi src(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi dest(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    for (int i = 0; i < U.size; i++)
    {
        U[i].data[0] = U_real[i];
        U[i].data[1] = U_imag[i];
    }
    for (int i = 0; i < src.size; i++)
    {
        src[i].data[0] = src_real[i];
        src[i].data[1] = src_imag[i];
    }
    dslash(U, src, dest, test);
    for (int i = 0; i < dest.size; i++)
    {
        dest_real[i] = dest[i].data[0];
        dest_imag[i] = dest[i].data[1];
    }
}
void ext_cg(const double *U_real, const double *U_imag, const double *b_real, const double *b_imag, double *x_real, double *x_imag, const int &lat_x, const int &lat_y, const int &lat_z, const int &lat_t, const int &lat_s, const int &lat_c, const int MAX_ITER, const double TOL, const bool &test)
{
    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi b(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi x(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    for (int i = 0; i < U.size; i++)
    {
        U[i].data[0] = U_real[i];
        U[i].data[1] = U_imag[i];
    }
    for (int i = 0; i < b.size; i++)
    {
        b[i].data[0] = b_real[i];
        b[i].data[1] = b_imag[i];
    }
    cg(U, b, x, MAX_ITER, TOL, test);
    for (int i = 0; i < x.size; i++)
    {
        x_real[i] = x[i].data[0];
        x_imag[i] = x[i].data[1];
    }
}
int main()
{
    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi src(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi dest(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi dest0(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    U.assign_random(666);
    src.assign_random(777);
    dest.assign_random(888);
    dslash(U, src, dest, false);
    // dest.print();
    return 0;
}