/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#define NO_IMPORT_ARRAY
#ifndef DISABLE_COOL_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL core_ARRAY_API
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "timer.h"

#include "timerWrapper.h"

///Allocate an aligned matrix of float with a pitch
/**	@param self : pointer to the parent object if it exist
 * 	@param args : arguments passed to the program
 * 	@return result of rdtsc function
*/
PyObject * timerWrapper(PyObject *self, PyObject *args){
	size_t res(rdtsc());
	return Py_BuildValue("k", res);
}

