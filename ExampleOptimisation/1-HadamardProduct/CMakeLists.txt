project(HadamardProduct)
cmake_minimum_required(VERSION 3.0)

add_executable(hadamard_product_O0 main.cpp)
set_property(TARGET hadamard_product_O0 PROPERTY COMPILE_FLAGS "-O0")
target_link_libraries(hadamard_product_O0 asterics_hpc)
runExample(hadamard_product_O0)

add_executable(hadamard_product_O1 main.cpp)
set_property(TARGET hadamard_product_O1 PROPERTY COMPILE_FLAGS "-O1")
target_link_libraries(hadamard_product_O1 asterics_hpc)
runExample(hadamard_product_O1)

add_executable(hadamard_product_O2 main.cpp)
set_property(TARGET hadamard_product_O2 PROPERTY COMPILE_FLAGS "-O2")
target_link_libraries(hadamard_product_O2 asterics_hpc)
runExample(hadamard_product_O2)

add_executable(hadamard_product_O3 main.cpp)
set_property(TARGET hadamard_product_O3 PROPERTY COMPILE_FLAGS "-O3")
target_link_libraries(hadamard_product_O3 asterics_hpc)
runExample(hadamard_product_O3)

add_executable(hadamard_product_Ofast main.cpp)
set_property(TARGET hadamard_product_Ofast PROPERTY COMPILE_FLAGS "-Ofast")
target_link_libraries(hadamard_product_Ofast asterics_hpc)
runExample(hadamard_product_Ofast)

plotPerf("hadamardBase" hadamard_product_O0 hadamard_product_O1 hadamard_product_O2 hadamard_product_O3 hadamard_product_Ofast)

add_executable(hadamard_product_vectorizeAVX main_vectorize.cpp)
set_property(TARGET hadamard_product_vectorizeAVX PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -mavx")
target_link_libraries(hadamard_product_vectorizeAVX asterics_hpc)
runExample(hadamard_product_vectorizeAVX)

add_executable(hadamard_product_vectorizeSSE main_vectorize.cpp)
set_property(TARGET hadamard_product_vectorizeSSE PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -msse")
target_link_libraries(hadamard_product_vectorizeSSE asterics_hpc)
runExample(hadamard_product_vectorizeSSE)

add_executable(hadamard_product_vectorizeMMX main_vectorize.cpp)
set_property(TARGET hadamard_product_vectorizeMMX PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -mmmx")
target_link_libraries(hadamard_product_vectorizeMMX asterics_hpc)
runExample(hadamard_product_vectorizeMMX)

add_executable(hadamard_product_vectorizeAVX2 main_vectorize.cpp)
set_property(TARGET hadamard_product_vectorizeAVX2 PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -mavx2")
target_link_libraries(hadamard_product_vectorizeAVX2 asterics_hpc)
runExample(hadamard_product_vectorizeAVX2)

plotPerf("hadamardVectorize" hadamard_product_O3 hadamard_product_vectorizeAVX hadamard_product_vectorizeSSE hadamard_product_vectorizeMMX hadamard_product_vectorizeAVX2)

add_executable(hadamard_product_intrinsics main_intrinsics.cpp)
set_property(TARGET hadamard_product_intrinsics PROPERTY COMPILE_FLAGS "-O3 -march=native -mtune=native")
target_link_libraries(hadamard_product_intrinsics asterics_hpc)
runExample(hadamard_product_intrinsics)

plotPerf("hadamardIntrinsics" hadamard_product_O3 hadamard_product_vectorizeAVX hadamard_product_intrinsics)
