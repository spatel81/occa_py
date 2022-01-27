#!/bin/bash

module purge
module load cmake
module load openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0
module load nvhpc ##For the build of nekBench
module load conda/2021-11-30

export OCCA_DIR=/grand/catalyst/spatel/OCCA_ML/occa/install/

export PATH+=":${OCCA_DIR}/bin"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"

export OCCA_CXX="g++"
export OCCA_CXXFLAGS="-O3"

#export OCCA_DPCPP_COMPILER="dpcpp"
#export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device 0x020a\""

export OCCA_CUDA_COMPILER="nvcc"
export OCCA_CUDA_COMPILER_FLAGS="-O3 --fmad=true"

#export EnableWalkerPartition=0
#export OCCA_CXXFLAGS="-O2"

# Run
# ./examples_cpp_add_vectors --device "{mode: 'CUDA', device_id: 0}"
