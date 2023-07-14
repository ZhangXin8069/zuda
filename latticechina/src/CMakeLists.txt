set(LATTICE_SOURCE
		${CMAKE_CURRENT_SOURCE_DIR}/lattice_propagator.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/lattice_fermion.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/lattice_gauge.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/dslash.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/dslash_gpu.cpp
		PARENT_SCOPE)

set(LATTICE_TARGET_HIP_SOURCE
		${CMAKE_CURRENT_SOURCE_DIR}/target/hip/DiracSetup.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/target/hip/DiracWilson.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/target/hip/DiracOverlapWilson.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/target/hip/legacyFunction.cpp
		PARENT_SCOPE)
