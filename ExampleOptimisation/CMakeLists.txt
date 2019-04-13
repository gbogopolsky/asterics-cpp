cmake_minimum_required(VERSION 3.0)
project(HPC_ASTERICS)

# Be sure that python3 exists (this might not be the case on mac)
execute_process(COMMAND which python3
		OUTPUT_VARIABLE PYTHON3_FROM_WHITCH)

# Then use the proprer python
if(PYTHON3_FROM_WHITCH)
	set(PYTHON_EXECUTABLE "python3" CACHE BOOL "Python program")
else()
	set(PYTHON_EXECUTABLE "python" CACHE BOOL "Python program")
endif()

add_subdirectory(Performances)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Examples/Performances)

include(runExample.cmake)
include(pythonCheck.cmake)

set(VECTOR_ALIGNEMENT 32)
add_definitions(-DVECTOR_ALIGNEMENT=${VECTOR_ALIGNEMENT})

add_subdirectory(AstericsHPC)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/AstericsHPC)

add_subdirectory(1-HadamardProduct)
