C_COMPILER          = gcc
CPP_C_COMPILER      = g++
MPI_COMPILER        =
MPI_LIB_PATH        = /usr/lib/x86_64-linux-gnu
MPI_INCLUDE_PATH    = /usr/include/mpi

DSLASH_SOURCE_PATH  = ./src
DSLASH_INCLUDE_PATH = ./inc
DSLASH_COMPILE_FLAG = -fPIC
DSLASH_LIBRARY_NAME = liblattice.so

${DSLASH_LIBRARY_NAME}: ${DSLASH_SOURCE_PATH}/dslash.o ${DSLASH_SOURCE_PATH}/lattice_fermion.o \
            ${DSLASH_SOURCE_PATH}/lattice_gauge.o ${DSLASH_SOURCE_PATH}/lattice_propagator.o ${DSLASH_SOURCE_PATH}/dslash_gpu.o
	${CPP_C_COMPILER} --shared -o $@ $^ -L${MPI_LIB_PATH} -lmpi -lmpi_cxx

${DSLASH_SOURCE_PATH}/dslash.o: ${DSLASH_SOURCE_PATH}/dslash.cpp ${DSLASH_INCLUDE_PATH}/operator.h
	${CPP_C_COMPILER} ${DSLASH_COMPILE_FLAG} -o $@ -c $< -I${MPI_INCLUDE_PATH} -I${DSLASH_INCLUDE_PATH}

${DSLASH_SOURCE_PATH}/dslash_gpu.o: ${DSLASH_SOURCE_PATH}/dslash_gpu.cpp ${DSLASH_INCLUDE_PATH}/operator.h
        ${CPP_C_COMPILER} ${DSLASH_COMPILE_FLAG} -o $@ -c $< -I${MPI_INCLUDE_PATH} -I${DSLASH_INCLUDE_PATH}

${DSLASH_SOURCE_PATH}/lattice_fermion.o: ${DSLASH_SOURCE_PATH}/lattice_fermion.cpp \
                                         ${DSLASH_INCLUDE_PATH}/lattice_fermion.h ${DSLASH_INCLUDE_PATH}/operator.h
	${CPP_C_COMPILER} ${DSLASH_COMPILE_FLAG} -o $@ -c $< -I${MPI_INCLUDE_PATH} -I${DSLASH_INCLUDE_PATH}

${DSLASH_SOURCE_PATH}/lattice_gauge.o: ${DSLASH_SOURCE_PATH}/lattice_gauge.cpp \
                                       ${DSLASH_INCLUDE_PATH}/lattice_gauge.h  \
                                       ${DSLASH_INCLUDE_PATH}/operator.h
	${CPP_C_COMPILER} ${DSLASH_COMPILE_FLAG} -o $@ -c $< -I${MPI_INCLUDE_PATH} -I${DSLASH_INCLUDE_PATH}

${DSLASH_SOURCE_PATH}/lattice_propagator.o: ${DSLASH_SOURCE_PATH}/lattice_propagator.cpp \
                                            ${DSLASH_INCLUDE_PATH}/lattice_propagator.h  \
                                            ${DSLASH_INCLUDE_PATH}/operator.h
	${CPP_C_COMPILER} ${DSLASH_COMPILE_FLAG} -o $@ -c $< -I${MPI_INCLUDE_PATH} -I${DSLASH_INCLUDE_PATH}

clean:
	rm -f ${DSLASH_SOURCE_PATH}/lattice_fermion.o ${DSLASH_SOURCE_PATH}/lattice_gauge.o \
	      ${DSLASH_SOURCE_PATH}/lattice_propagator.o ${DSLASH_SOURCE_PATH}/dslash.o \
	      ${DSLASH_LIBRARY_NAME}
