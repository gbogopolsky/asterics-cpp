project(AstericsHPC)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_asterics_hpc_module setup.py ${moduleSrc})

add_dependencies(install_asterics_hpc_module asterics_hpc)

