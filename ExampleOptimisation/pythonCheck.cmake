if(DEFINED ENV{CONDA_DEFAULT_ENV})
	message(STATUS "ENV{CONDA_DEFAULT_ENV} exist $ENV{CONDA_DEFAULT_ENV}")
	message(STATUS "ENV{CONDA_PREFIX} exist $ENV{CONDA_PREFIX}")
	message(STATUS "ENV{CONDA_ENV_PATH} = $ENV{CONDA_ENV_PATH}")
	
	if(DEFINED ENV{CONDA_ENV_PATH})
		set(PYTHON_LIBRARY_DIR $ENV{CONDA_ENV_PATH}/lib CACHE STRING "link directory of python")
	endif()
	if(DEFINED ENV{CONDA_PREFIX})
		set(PYTHON_LIBRARY_DIR $ENV{CONDA_PREFIX}/lib CACHE STRING "link directory of python")
	endif()
	set(PYTHON_INSTALL_PREFIX "" CACHE STRING "Install prefix of the python plib functions")
else()
	message(STATUS "ENV{CONDA_DEFAULT_ENV} does not exist")
	set(PYTHON_INSTALL_PREFIX $ENV{HOME}/.local CACHE STRING "Install prefix of the python plib functions")
endif()

if(PYTHON_INSTALL_PREFIX)
	set (PYTHON_INSTALL_PREFIX "--prefix=${PYTHON_INSTALL_PREFIX}")
endif()

set(ASTERICS_HPC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/AstericsHPC)
set(ASTERICS_HPC_PYINCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/AstericsHPC/astericshpc)
set(ASTERICS_CPP_LIBRARY_BUILD ${CMAKE_CURRENT_BINARY_DIR}/AstericsHPC)
set(ASTERICS_CPP_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(SCRIPT_CALL_PYTHON_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/AstericsHPC/astericshpc/scriptCallPythonSetup.sh.cmake)

#Create a python module during the build
# 	targetName : name of the target to be created
# 	setupFile : setup.py file to be used
# 	moduleSrc : source python, C, C++ used to create the module
function(createPythonModule targetName setupFile moduleSrc)
	configure_file(${setupFile} ${CMAKE_CURRENT_BINARY_DIR}/setup.py @ONLY)
	configure_file(${SCRIPT_CALL_PYTHON_SETUP} ${CMAKE_CURRENT_BINARY_DIR}/scriptCallPythonSetup.sh @ONLY)
	add_custom_command(
		OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist
		COMMAND ${CMAKE_CURRENT_BINARY_DIR}/scriptCallPythonSetup.sh
		COMMENT "Install ${targetName} python module"
		DEPENDS ${moduleSrc}
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	)
	add_custom_target("${targetName}" ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dist)
endfunction(createPythonModule)

