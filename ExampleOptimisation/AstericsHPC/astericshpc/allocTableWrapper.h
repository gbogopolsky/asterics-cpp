/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#ifndef __ALLOC_TABLE_WRAPPER_H__
#define __ALLOC_TABLE_WRAPPER_H__

#include <Python.h>
#include "structmember.h"

PyObject * allocTable(long unsigned int nbElement);
PyObject * allocTableWrapper(PyObject *self, PyObject *args);

#endif
