project(AstericsHPC)
cmake_minimum_required(VERSION 3.0)

add_library(asterics_hpc SHARED asterics_alloc.cpp timer.cpp)
set_property(TARGET asterics_hpc PROPERTY COMPILE_FLAGS "-O3")

# install(TARGETS asterics_hpc LIBRARY DESTINATION lib)

add_subdirectory(astericshpc)

