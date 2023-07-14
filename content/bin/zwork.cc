#include <complex>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

using namespace std;

class lattice_fermi
{
public:
    int lat_x, lat_t, lat_spin;
    int size;
    vector<complex<double>> lattice_vec;
    lattice_fermi(int lat_x, int lat_t, int lat_spin) : lat_x(lat_x), lat_t(lat_t), lat_spin(lat_spin), lattice_vec(lat_x * lat_t * lat_spin)
    {
        size = lattice_vec.size();
    }
    lattice_fermi();
    void clean()
    {
        for (int i = 0; i < size; i++)
            lattice_vec[i] = 0;
    }
    complex<double> &operator[](int i) // const
    {
        return lattice_vec[i];
    }
    lattice_fermi &operator=(lattice_fermi &a)
    {
        for (int i = 0; i < size; i++)
            this->lattice_vec[i] = a.lattice_vec[i];
        return *this;
    }
    lattice_fermi &operator-(const lattice_fermi &a)
    {
        for (int i = 0; i < size; i++)
            this->lattice_vec[i] = this->lattice_vec[i] - a.lattice_vec[i];
        return *this;
    }

    lattice_fermi &operator+(const lattice_fermi &a)
    {
        for (int i = 0; i < size; i++)
            this->lattice_vec[i] = this->lattice_vec[i] + a.lattice_vec[i];
        return *this;
    }
};

class lattice_gauge
{
public:
    int lat_x, lat_t, lat_d;
    int size;
    vector<complex<double>> lattice_vec_c;
    lattice_gauge(int lat_x, int lat_t, int lat_d) : lat_x(lat_x), lat_t(lat_t), lat_d(lat_d), lattice_vec_c(lat_x * lat_t * lat_d)
    {
        size = lattice_vec_c.size();
    }
    lattice_gauge();
    complex<double> &operator[](int i) // const
    {
        return lattice_vec_c[i];
    }
    lattice_gauge &operator=(lattice_gauge a)
    {

        for (int i = 0; i < size; i++)
            this->lattice_vec_c[i] = a.lattice_vec_c[i];
        return *this;
    }
};

class lattice_propagator
{
public:
    int lat_x, lat_t, lat_spin;
    int size;
    vector<complex<double>> lattice_vec_c;
    lattice_propagator(int lat_x, int lat_t, int lat_spin) : lat_x(lat_x), lat_t(lat_t), lat_spin(lat_spin), lattice_vec_c(lat_x * lat_t * lat_spin * lat_spin)
    {
        size = lattice_vec_c.size();
    }
    //  lattice_propagator(lattice_propagator & prop):lat_x(prop.lat_x),lat_t(prop.lat_t),lat_spin(prop.lat_spin), lattice_vec_c(prop.lat_x*prop.lat_t*prop.lat_spin*prop.lat_spin)
    // {
    //      size = lattice_vec_c.size();
    // }
    lattice_propagator();
    void clean()
    {
        for (int i = 0; i < size; i++)
            lattice_vec_c[i] = 0;
    }
    complex<double> &operator[](int i) // const
    {
        return lattice_vec_c[i];
    }
    lattice_propagator &operator=(lattice_propagator a)
    {

        for (int i = 0; i < size; i++)
            this->lattice_vec_c[i] = a.lattice_vec_c[i];
        return *this;
    }
};

class Gamma
{
public:
    int lat_spin;
    int size;
    vector<complex<double>> lattice_vec_c;
    Gamma(int lat_spin) : lat_spin(lat_spin), lattice_vec_c(lat_spin * lat_spin)
    {
        size = lattice_vec_c.size();
    }
    Gamma();
    void clean()
    {
        for (int i = 0; i < size; i++)
            lattice_vec_c[i] = 0;
    }
    complex<double> &operator[](int i) // const
    {
        return lattice_vec_c[i];
    }
    Gamma &operator=(Gamma a)
    {

        for (int i = 0; i < size; i++)
            this->lattice_vec_c[i] = a.lattice_vec_c[i];
        return *this;
    }
};

lattice_propagator operator*(Gamma G, lattice_propagator prop)

{
    lattice_propagator prop1(prop);
    prop1.clean();
    for (int x = 0; x < prop.lat_x; x++)
        for (int t = 0; t < prop.lat_t; t++)
            for (int s1 = 0; s1 < prop.lat_spin; s1++)
                for (int s2 = 0; s2 < prop.lat_spin; s2++)
                {
                    for (int i = 0; i < prop.lat_spin; i++)
                        prop1[x * prop.lat_t * prop.lat_spin * prop.lat_spin + t * prop.lat_spin * prop.lat_spin + s1 * prop.lat_spin + s2] +=
                            G[s1 * G.lat_spin + i] * prop[x * prop.lat_t * prop.lat_spin * prop.lat_spin + t * prop.lat_spin * prop.lat_spin + i * prop.lat_spin + s2];
                }
    return prop1;
}

lattice_propagator operator*(lattice_propagator prop, Gamma G)

{
    lattice_propagator prop1(prop);
    prop1.clean();
    for (int x = 0; x < prop.lat_x; x++)
        for (int t = 0; t < prop.lat_t; t++)
            for (int s1 = 0; s1 < prop.lat_spin; s1++)
                for (int s2 = 0; s2 < prop.lat_spin; s2++)
                {
                    for (int i = 0; i < prop.lat_spin; i++)
                        prop1[x * prop.lat_t * prop.lat_spin * prop.lat_spin + t * prop.lat_spin * prop.lat_spin + s1 * prop.lat_spin + s2] +=
                            G[i * G.lat_spin + s2] * prop[x * prop.lat_t * prop.lat_spin * prop.lat_spin + t * prop.lat_spin * prop.lat_spin + s1 * prop.lat_spin + i];
                }
    return prop1;
}

double norm_2(lattice_fermi s)
{
    complex<double> s1(0.0, 0.0);
    double tmp;
    for (int i = 0; i < s.size; i++)
    {
        s1 += s[i] * conj(s[i]);
    }
    tmp = s1.real();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&tmp, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return tmp;
};

double norm_2(lattice_propagator f)
{

    complex<double> f1(0.0, 0.0);
    double tmp;
    for (int i = 0; i < f.size; i++)
    {
        f1 += f[i] * conj(f[i]);
        // printf("s=%f\n", f[i].real());
    }
    tmp = f1.real();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&tmp, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return tmp;
};

complex<double> vector_p(lattice_fermi r1, lattice_fermi r2)
{
    complex<double> ro(0.0, 0.0), tmp;
    for (int i = 0; i < r1.size; i++)
    {
        ro += (conj(r1[i]) * r2[i]);
    }
    tmp = ro;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&tmp, &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    return tmp;
};

void Dslash2(lattice_fermi src, lattice_fermi &dest, lattice_gauge U, const double mass, const bool dag)
{
    int node_size, node_rank, backward_rank, forward_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    backward_rank = (node_rank + node_size - 1) % node_size;
    forward_rank = (node_rank + 1) % node_size;
    std::complex<double> *send_vec;
    send_vec = new std::complex<double>[src.lat_t];
    std::complex<double> *recv_vec;
    recv_vec = new std::complex<double>[src.lat_t];
    dest.clean();
    const double a = 2.0;
    const complex<double> i(0.0, 1.0);
    complex<double> tmp;
    const double Half = 0.5;
    double flag = (dag == true) ? -1 : 1;
    for (int x = 0; x < src.lat_x; x++)
    {
        if (x == 0)
        {
            for (int t = 0; t < src.lat_t; t++)
            {

                // mass term
                for (int s = 0; s < src.lat_spin; s++)
                {
                    dest[(x * src.lat_t + t) * 2 + s] += -(a + mass) * src[(x * src.lat_t + t) * 2 + s];
                }
                // backward x
                int b_x = (x + src.lat_x - 1) % src.lat_x;
                send_vec[t] = (src[(x * src.lat_t + t) * 2 + 0] + flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(b_x * src.lat_t + t) * 2 + 0];
                // dest[(b_x * src.lat_t + t) * 2 + 0] += tmp;
                // dest[(b_x * src.lat_t + t) * 2 + 1] += flag * tmp;

                // forward x
                int f_x = (x + 1) % src.lat_x;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] - flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 0]);
                dest[(f_x * src.lat_t + t) * 2 + 0] += tmp;
                dest[(f_x * src.lat_t + t) * 2 + 1] -= flag * tmp;

                // backward t
                int b_t = (t + src.lat_t - 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] + flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(x * src.lat_t + b_t) * 2 + 1];
                dest[(x * src.lat_t + b_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + b_t) * 2 + 1] -= flag * i * tmp;

                // forward t
                int f_t = (t + 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] - flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 1]);
                dest[(x * src.lat_t + f_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + f_t) * 2 + 1] += flag * i * tmp;
            }
            MPI_Request send_request;
            MPI_Isend(send_vec, src.lat_t, MPI_DOUBLE_COMPLEX, backward_rank, backward_rank, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            // std::cout << "send to backward:Rank# " << node_rank << "->Rank# " << backward_rank << std::endl;

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Request recv_request;
            MPI_Irecv(recv_vec, src.lat_t, MPI_DOUBLE_COMPLEX, backward_rank, node_rank, MPI_COMM_WORLD, &recv_request);
            // MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int t = 0; t < src.lat_t; t++)
            {
                dest[(x * src.lat_t + t) * 2 + 0] += recv_vec[t];
                dest[(x * src.lat_t + t) * 2 + 1] -= flag * recv_vec[t];
            }
            // std::cout << "recv from backward:Rank# " << node_rank << "<-Rank# " << backward_rank << std::endl;
        }
        else if (x == src.lat_x - 1)
        {

            for (int t = 0; t < src.lat_t; t++)
            {
                // mass term
                for (int s = 0; s < src.lat_spin; s++)
                {
                    dest[(x * src.lat_t + t) * 2 + s] += -(a + mass) * src[(x * src.lat_t + t) * 2 + s];
                }

                // backward x
                int b_x = (x + src.lat_x - 1) % src.lat_x;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] + flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(b_x * src.lat_t + t) * 2 + 0];
                dest[(b_x * src.lat_t + t) * 2 + 0] += tmp;
                dest[(b_x * src.lat_t + t) * 2 + 1] += flag * tmp;

                // forward x
                int f_x = (x + 1) % src.lat_x;
                send_vec[t] = (src[(x * src.lat_t + t) * 2 + 0] - flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 0]);
                // dest[(f_x * src.lat_t + t) * 2 + 0] += tmp;
                // dest[(f_x * src.lat_t + t) * 2 + 1] -= flag * tmp;

                // backward t
                int b_t = (t + src.lat_t - 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] + flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(x * src.lat_t + b_t) * 2 + 1];
                dest[(x * src.lat_t + b_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + b_t) * 2 + 1] -= flag * i * tmp;

                // forward t
                int f_t = (t + 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] - flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 1]);
                dest[(x * src.lat_t + f_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + f_t) * 2 + 1] += flag * i * tmp;
            }
            MPI_Request send_request;
            MPI_Isend(send_vec, src.lat_t, MPI_DOUBLE_COMPLEX, forward_rank, forward_rank, MPI_COMM_WORLD, &send_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            // std::cout << "send to forward:Rank# " << node_rank << "->Rank# " << forward_rank << std::endl;

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Request recv_request;
            MPI_Irecv(recv_vec, src.lat_t, MPI_DOUBLE_COMPLEX, forward_rank, node_rank, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int t = 0; t < src.lat_t; t++)
            {
                dest[(x * src.lat_t + t) * 2 + 0] += recv_vec[t];
                dest[(x * src.lat_t + t) * 2 + 1] += flag * recv_vec[t];
            }
            // std::cout << "recv from forward:Rank# " << node_rank << "<-Rank# " << forward_rank << std::endl;
        }
        else
        {
            for (int t = 0; t < src.lat_t; t++)
            {

                // mass term
                for (int s = 0; s < src.lat_spin; s++)
                {
                    dest[(x * src.lat_t + t) * 2 + s] += -(a + mass) * src[(x * src.lat_t + t) * 2 + s];
                }

                // backward x
                int b_x = (x + src.lat_x - 1) % src.lat_x;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] + flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(b_x * src.lat_t + t) * 2 + 0];
                dest[(b_x * src.lat_t + t) * 2 + 0] += tmp;
                dest[(b_x * src.lat_t + t) * 2 + 1] += flag * tmp;

                // forward x
                int f_x = (x + 1) % src.lat_x;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] - flag * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 0]);
                dest[(f_x * src.lat_t + t) * 2 + 0] += tmp;
                dest[(f_x * src.lat_t + t) * 2 + 1] -= flag * tmp;

                // backward t
                int b_t = (t + src.lat_t - 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] + flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * U[(x * src.lat_t + b_t) * 2 + 1];
                dest[(x * src.lat_t + b_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + b_t) * 2 + 1] -= flag * i * tmp;

                // forward t
                int f_t = (t + 1) % src.lat_t;
                tmp = (src[(x * src.lat_t + t) * 2 + 0] - flag * i * src[(x * src.lat_t + t) * 2 + 1]) * Half * conj(U[(x * src.lat_t + t) * 2 + 1]);
                dest[(x * src.lat_t + f_t) * 2 + 0] += tmp;
                dest[(x * src.lat_t + f_t) * 2 + 1] += flag * i * tmp;
            }
        }
    }
    delete[] send_vec;
    delete[] recv_vec;
}

void fermi_to_prop(lattice_fermi dest, lattice_propagator &prop, int i)
{
    for (int x = 0; x < dest.lat_x; x++)
        for (int t = 0; t < dest.lat_t; t++)
            for (int s = 0; s < dest.lat_spin; s++)
            {
                prop[(x * prop.lat_t + t) * prop.lat_spin * prop.lat_spin + (i * prop.lat_spin) + s] = dest[(x * prop.lat_t + t) * prop.lat_spin + s];
                //    //printf("dest=%f\n",dest[(x*prop.lat_t+t)*prop.lat_spin+s].real());
                //    //printf("dest=%f\n",prop[(x*prop.lat_t+t)*prop.lat_spin*prop.lat_spin+(i*prop.lat_spin)+s].real());
            }
}

int CG(lattice_fermi src, lattice_fermi &dest, lattice_gauge U, const double mass, const int max)
{
    lattice_fermi r0(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi rr0(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi z0(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi r1(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi z1(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi q(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi qq(src.lat_x, src.lat_t, src.lat_spin);
    lattice_fermi P(src.lat_x, src.lat_t, src.lat_spin);
    complex<double> aphi;
    complex<double> beta;
    for (int i = 0; i < src.size; i++)
    {
        r0[i] = 0;
    }
    for (int i = 0; i < dest.size; i++)
    {
        dest[i] = 0;
    }

    Dslash2(dest, rr0, U, mass, false);
    Dslash2(rr0, r0, U, mass, true);
    for (int i = 0; i < src.size; i++)
    {
        r0[i] = src[i] - r0[i];
    }
    for (int f = 1; f < max; f++)
    {

        // printf("f=%i\n", f);
        std::complex<double> rho;
        rho = vector_p(r0, r0);

        std::complex<double> rho1;
        rho1 = vector_p(r1, r1);
        for (int i = 0; i < z0.size; i++)
        {
            z0[i] = r0[i];
        }
        if (f == 1)
        {
            for (int i = 0; i < P.size; i++)
            {
                P[i] = z0[i];
            }
            // printf("P=%f\n", norm_2(P));
        }
        else
        {
            beta = rho / rho1;
            // printf("beta=%f\n", beta);
            for (int i = 0; i < P.size; i++)
                P[i] = z0[i] + beta * P[i];
            // printf("P=%f\n", norm_2(P));
        }
        Dslash2(P, qq, U, mass, false);
        // printf("d_qq=%f\n", norm_2(qq));
        Dslash2(qq, q, U, mass, true);
        // printf("q=%f\n", norm_2(q));
        aphi = rho / vector_p(P, q);
        // printf("d_q=%f\n", vector_p(P, q));
        // printf("aphi=%f\n", aphi);
        for (int i = 0; i < dest.size; i++)
            dest[i] = dest[i] + aphi * P[i];
        for (int i = 0; i < r1.size; i++)
            r1[i] = r0[i];
        for (int i = 0; i < r0.size; i++)
            r0[i] = r0[i] - aphi * q[i];
        if (norm_2(r0) < 1e-12 || f == max - 1)
        {
            // printf("convergence recedul=1e-5\n");
            std::cout
                << "##########"
                << "loop cost:"
                << f
                << std::endl
                << "convergence recedul:"
                << norm_2(r0)
                << std::endl;
            break;
        }
    } // for (f) end

    return 0;

} // CG func end

int main()
{

    MPI_Init(NULL, NULL);
    double start, end;
    int node_size, node_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &node_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_rank);
    start = MPI_Wtime();
    // gird distance
    int nx = 100;
    nx /= node_size;
    int nt = 100;
    int ns = 2;
    int nd = 2;
    double mass = 1;

    lattice_fermi src(nx, nt, ns);
    lattice_fermi ssrc(nx, nt, ns);
    lattice_fermi dest(nx, nt, ns);
    lattice_fermi dest_1(nx, nt, ns);
    lattice_fermi src1(nx, nt, ns);
    lattice_gauge U(nx, nt, ns);
    lattice_propagator prop(nx, nt, ns);
    for (int i = 0; i < U.size; i++)
    {
        U[i] = 1.0;
    }
    for (int i = 0; i < src.size; i++)
    {
        if (i == 0 && node_rank == 0)
            src[i] = 1;
        else
            src[i] = 0;
    }
    // for (int i = 0; i < src1.size; i++)
    //     if (i == 1)
    //         src1[i] = 1;
    //     else
    //         src1[i] = 0;

    Dslash2(src, ssrc, U, mass, true);
    CG(ssrc, dest, U, mass, 1000);
    // Dslash2(src1, ssrc, U, mass, true);
    // CG(ssrc, dest_1, U, mass, 1000);
    // Dslash2(dest,ssrc,U,mass,false);
    fermi_to_prop(dest, prop, 0);
    // fermi_to_prop(dest_1,prop,1);

    // for (int x = 0; x < dest.lat_x; x++)
    //     for (int t = 0; t < dest.lat_t; t++)
    //         for (int s = 0; s < dest.lat_spin; s++)
    //         {
    //             //    //printf("dest=%f\n",prop[(x*prop.lat_t+t)*prop.lat_spin*prop.lat_spin+(0*prop.lat_spin)+s].real());
    //         }

    // //printf("s1=%f\n",s1.real());

    // //printf("norm_propagator0=%f\n",norm_2(dest));
    printf("norm_propagator1=%f\n", norm_2(prop));
    // //printf("norm_src-propagator=%.10e\n",norm_2(ssrc-src));
    // //printf("dslash_1=%f\n",norm_2(ssrc));
    // //printf("dslash_2=%f\n",norm_2(dest));
    end = MPI_Wtime();
    std::cout
        << "##########"
        << "time cost:"
        << end - start
        << "s"
        << std::endl;
    MPI_Finalize();
    return 0;
}
