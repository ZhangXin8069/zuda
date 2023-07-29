#include "../include/zuda_cpu.h"
int main(){
    int lat_x(16);
    int lat_y(16);
    int lat_z(16);
    int lat_t(32);
    int lat_s(4);
    int lat_c(3);
    //int MAX_ITER(1e6);
    //double TOL(1e-6);
    LatticeGauge U(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi src(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    LatticeFermi dest(lat_x, lat_y, lat_z, lat_t, lat_s, lat_c);
    U.assign_random();
    src.assign_random();
    dest.assign_zero();
    clock_t start = clock();
    dslash(U,src,dest,false);
    clock_t end = clock();
    std::cout
        << "################"
        << "time cost:"
        << (double)(end - start) / CLOCKS_PER_SEC
        << "s"
        << std::endl;
    return 0;
}