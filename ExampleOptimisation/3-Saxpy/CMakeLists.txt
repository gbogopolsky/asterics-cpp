project(Saxpy)
cmake_minimum_required(VERSION 3.0)

add_executable(saxpy_O0 main.cpp)
set_property(TARGET saxpy_O0 PROPERTY COMPILE_FLAGS "-O0")
target_link_libraries(saxpy_O0 asterics_hpc)
runExample(saxpy_O0)

add_executable(saxpy_O1 main.cpp)
set_property(TARGET saxpy_O1 PROPERTY COMPILE_FLAGS "-O1")
target_link_libraries(saxpy_O1 asterics_hpc)
runExample(saxpy_O1)

add_executable(saxpy_O2 main.cpp)
set_property(TARGET saxpy_O2 PROPERTY COMPILE_FLAGS "-O2")
target_link_libraries(saxpy_O2 asterics_hpc)
runExample(saxpy_O2)

add_executable(saxpy_O3 main.cpp)
set_property(TARGET saxpy_O3 PROPERTY COMPILE_FLAGS "-O3")
target_link_libraries(saxpy_O3 asterics_hpc)
runExample(saxpy_O3)

add_executable(saxpy_Ofast main.cpp)
set_property(TARGET saxpy_Ofast PROPERTY COMPILE_FLAGS "-Ofast")
target_link_libraries(saxpy_Ofast asterics_hpc)
runExample(saxpy_Ofast)

plotPerf("saxpyBase" saxpy_O0 saxpy_O1 saxpy_O2 saxpy_O3 saxpy_Ofast) 

add_executable(saxpy_vectorizeAVX main_vectorize.cpp)
set_property(TARGET saxpy_vectorizeAVX PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -mavx")
target_link_libraries(saxpy_vectorizeAVX asterics_hpc)
runExample(saxpy_vectorizeAVX)

add_executable(saxpy_vectorizeSSE main_vectorize.cpp)
set_property(TARGET saxpy_vectorizeSSE PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -msse")
target_link_libraries(saxpy_vectorizeSSE asterics_hpc)
runExample(saxpy_vectorizeSSE)

add_executable(saxpy_vectorizeMMX main_vectorize.cpp)
set_property(TARGET saxpy_vectorizeMMX PROPERTY COMPILE_FLAGS "-O3 -ftree-vectorize -march=native -mtune=native -mmmx")
target_link_libraries(saxpy_vectorizeMMX asterics_hpc)
runExample(saxpy_vectorizeMMX)

plotPerf("saxpyVectorize" saxpy_O3 saxpy_vectorizeAVX saxpy_vectorizeSSE saxpy_vectorizeMMX)

add_executable(saxpy_intrinsics main_intrinsics.cpp)
set_property(TARGET saxpy_intrinsics PROPERTY COMPILE_FLAGS "-O3 -march=native -mtune=native")
target_link_libraries(saxpy_intrinsics asterics_hpc)
runExample(saxpy_intrinsics)

plotPerf("saxpyIntrinsics" saxpy_O3 saxpy_vectorizeAVX saxpy_intrinsics)