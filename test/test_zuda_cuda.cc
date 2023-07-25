#include "../include/zuda_cpuX.h"
int main(int argc, char **argv)
{
    double start, end;
    int lat_x(16), lat_y(16), lat_z(16), lat_t(32), lat_s(4), lat_c(3);
    int num_x(1), num_y(1), num_z(1), num_t(2);
    int MAX_ITER(1e6);
    double TOL(1e-12);
    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi b(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi x(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    U.assign_random(666);
    b.assign_zero();
    b(0, 0, 0, 0, 0, 0) = 1000.0;
    x.assign_zero();
    MPI_Init(&argc, &argv);
    LatticeGauge block_U = U.block(num_x, num_y, num_z, num_t);
    LatticeFermi block_b = b.block(num_x, num_y, num_z, num_t);
    LatticeFermi block_x = x.block(num_x, num_y, num_z, num_t);
    start = MPI_Wtime();
    cg(block_U, block_b, block_x, num_x, num_y, num_z, num_t, MAX_ITER, TOL, true);
    // dslash(block_U, block_b, block_x, false);
    end = MPI_Wtime();
    LatticeFermi reback_x = block_x.reback(num_x, num_y, num_z, num_t);
    // reback_x.print(0, 0, 0, 0, 0, 0);
    // reback_x.print(0, 0, 0, 0, 0, 1);
    std::cout
        << "######x.norm2():" << reback_x.norm_2() << "######" << std::endl;
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if(rank==0){
    // x.print();
    // }
    std::cout
        << "######time cost:" << end - start << "s ######" << std::endl;
    MPI_Finalize();
    return 0;
}
