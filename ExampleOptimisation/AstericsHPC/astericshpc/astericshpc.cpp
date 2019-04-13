/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifndef DISABLE_COOL_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL core_ARRAY_API
#endif

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include <string>

#include "allocTableWrapper.h"
#include "allocMatrixWrapper.h"
#include "timerWrapper.h"

std::string allocTable_docstring = "Allocate a table of float with a padding\n\
Parameters :\n\
	nbElement : number of elements of the table\n\
Return :\n\
	1 dimentional aligned numpy array initialised to 0";

std::string allocMatrix_docstring = "Allocate a matrix of float with a pitch\n\
Parameters :\n\
	nbRow : number of rows of the matrix\n\
	nbCol : number of colmuns of the matrix\n\
Return :\n\
	2 dimentional aligned numpy array initialised to 0 with a pitch";

std::string timerWrapper_docString = "Get the number of cycles since the begining of the program\n\
Return :\n\
	number of cycles since the begining of the program in uint64";

static PyMethodDef _astericshpc_methods[] = {
	{"allocTable", (PyCFunction)allocTableWrapper, METH_VARARGS, allocTable_docstring.c_str()},
	{"allocMatrix", (PyCFunction)allocMatrixWrapper, METH_VARARGS, allocMatrix_docstring.c_str()},
	{"rdtsc", (PyCFunction)timerWrapper, METH_NOARGS, timerWrapper_docString.c_str()},

	{NULL, NULL}
};

static PyModuleDef _astericshpc_module = {
	PyModuleDef_HEAD_INIT,
	"astericshpc",
	"",
	-1,
	_astericshpc_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module astericshpc
/**	@return python module astericshpc
*/
PyMODINIT_FUNC PyInit_astericshpc(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_astericshpc_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

