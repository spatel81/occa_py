project(perf)

add_executable(perf main.cpp)

include_directories(/grand/catalyst/spatel/OCCA_ML/occa/src)

include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/lib/python3.8/site-packages/tensorflow/include/external/local_config_python/numpy_include)
include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/include)
include_directories(/lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/include/python3.8)
include_directories(/soft/hpc-sdk/Linux_x86_64/21.7/cuda/include)

find_library(PYTHONLIB python3.8 HINTS /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/lib)

target_include_directories(perf PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(perf 
  PUBLIC
    MPI::MPI_C 
    libocca
    ${PYTHONLIB}
)

install(TARGETS perf RUNTIME DESTINATION ".")
install(DIRECTORY kernel/ DESTINATION kernel_perf)
install(FILES python_module.py DESTINATION ".")

##compile_cpp_example_with_modes(add_vectors main.cpp)
##add_custom_target(cpp_example_add_vectors_okl ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/addVectors.okl addVectors.okl)
##add_dependencies(examples_cpp_add_vectors cpp_example_add_vectors_okl)
