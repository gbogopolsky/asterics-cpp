/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#ifndef __ALLOCMATRIXWRAPPER_H__
#define __ALLOCMATRIXWRAPPER_H__

#include <Python.h>
#include "structmember.h"

PyObject * allocMatrix(long unsigned int nbRow, long unsigned int nbCol);
PyObject * allocMatrixWrapper(PyObject *self, PyObject *args);

#endif
