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

#include <Python.h>
#include <numpy/arrayobject.h>
#include <bytearrayobject.h>

#include "asterics_alloc.h"

#include "allocTableWrapper.h"

///Free the capsule memory
/**	@param obj : object with contains the capsule
*/
void empty_freeTabArray(PyObject* obj){
	float* ptr = (float*) PyCapsule_GetPointer(obj,"emptyTable");
	free(ptr);
}

///Create a numpy matrix
/**	@param nbElement : number of element of the table
 * 	@return numpy array
*/
PyObject * allocTable(long unsigned int nbElement){
	//Set the size of the numpy array
	npy_intp attr_size[2];
	attr_size[0] = nbElement;
	
	long unsigned int pitch(getPitch(nbElement));
	//Calling allocation
	float* tab = (float*)asterics_malloc((nbElement + pitch)*sizeof(float));
	if(tab == NULL){
		PyObject* objMat = PyArray_EMPTY(2, attr_size, NPY_FLOAT32, 0);
		if(objMat == NULL){
			PyErr_SetString(PyExc_RuntimeError, "allocTableWrapper : Could not allocated memory\n");
			return NULL;
		}
		return objMat;
	}
	memset(tab, 0, (nbElement + pitch)*sizeof(float));
	PyArray_Dims strides = {NULL, 0};
	strides.ptr = PyDimMem_NEW(1);
	strides.len = 1;
	PyArray_Descr *descr = PyArray_DescrFromType(NPY_FLOAT32);
	strides.ptr[0] = (npy_intp)sizeof(float);		     // Last strides is equal to element size
	
	PyObject* objMat = PyArray_NewFromDescr(&PyArray_Type, descr, 1, attr_size, strides.ptr, (void *)tab, NPY_ARRAY_WRITEABLE, NULL);
	
	//Desalocation stuff
	PyObject* memory_capsule = PyCapsule_New(tab, "emptyTable", empty_freeTabArray);
	if(PyArray_SetBaseObject((PyArrayObject*)objMat, memory_capsule) < 0){
		PyErr_SetString(PyExc_RuntimeError, "Fail to create PyCapsule\n");
		return NULL;
	}
	return objMat;
}

///Allocate an aligned matrix of float with a pitch
/**	@param self : pointer to the parent object if it exist
 * 	@param args : arguments passed to the program
 * 	@return allocated numpy array
*/
PyObject * allocTableWrapper(PyObject *self, PyObject *args){
	long unsigned int nbElement(0lu);
	if(!PyArg_ParseTuple(args, "k", &nbElement)){
		PyErr_SetString(PyExc_RuntimeError, "allocTableWrapper : wrong set of arguments. Expects one argument table size\n");
		return NULL;
	}
	return allocTable(nbElement);
}

