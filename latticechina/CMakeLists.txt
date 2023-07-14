cmake_minimum_required(VERSION 3.13.4)
project(lattice)

option (USE_MYMATH "Use provided math implementation" ON)

find_package(MPI REQUIRED)

message(STATUS "Using MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "Using MPI_C_LIBRARIES: ${MPI_C_LIBRARIES}")

if (NOT DEFINED ROCM_PATH )
    if (NOT DEFINED ENV{ROCM_PATH} )
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm path")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm path")
    endif()
endif()

list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm ${ROCM_PATH}/hip )
find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)

set(CMAKE_C_COMPILER ${ROCM_PATH}/hip/bin/hipcc)
set(CMAKE_CXX_COMPILER ${ROCM_PATH}/hip/bin/hipcc)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
message(Set CXX Compiler: ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories("${PROJECT_SOURCE_DIR}/inc")
include_directories("${ROCM_PATH}/hip/include")
add_subdirectory (src)

#add_executable(lattice dslashMain.cpp
#                       ${LATTICE_SOURCE})

add_library(lattice SHARED ${LATTICE_SOURCE}
                           ${LATTICE_TARGET_HIP_SOURCE} inc/target/hip/dslash_main_overlap_kernel.h)

target_link_libraries(lattice ${MPI_LIBRARIES} hip::host)


