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

#include "allocMatrixWrapper.h"

///Free the capsule memory
/**	@param obj : object with contains the capsule
*/
void empty_freeArray(PyObject* obj){
	float* ptr = (float*) PyCapsule_GetPointer(obj,"emptyMatrix");
	free(ptr);
}

///Create a numpy matrix
/**	@param nbRow : number of rows of the matrix
 * 	@param nbCol : number of columns of the matrix
 * 	@return numpy array
*/
PyObject * allocMatrix(long unsigned int nbRow, long unsigned int nbCol){
	//Set the size of the numpy array
	npy_intp attr_size[2];
	attr_size[0] = nbRow;
	attr_size[1] = nbCol;
	
	float* mat = asterics_malloc2f(nbRow, nbCol);
	if(mat == NULL){
		PyObject* objMat = PyArray_EMPTY(2, attr_size, NPY_FLOAT32, 0);
		if(objMat == NULL){
			PyErr_SetString(PyExc_RuntimeError, "allocMatrix : Could not allocated memory\n");
			return NULL;
		}
		return objMat;
	}
	long unsigned int pitch(getPitch(nbCol));
	
	PyArray_Dims strides = {NULL, 0};
	strides.ptr = PyDimMem_NEW(2);
	strides.len = 2;
	PyArray_Descr *descr = PyArray_DescrFromType(NPY_FLOAT32);
	strides.ptr[1] = (npy_intp)sizeof(float);		     // Last strides is equal to element size
	strides.ptr[0] = (pitch + nbCol) *  strides.ptr[1];
	
	PyObject* objMat = PyArray_NewFromDescr(&PyArray_Type, descr, 2, attr_size, strides.ptr, (void *)mat, NPY_ARRAY_WRITEABLE, NULL);
	
	//Desalocation stuff
	PyObject* memory_capsule = PyCapsule_New(mat, "emptyMatrix", empty_freeArray);
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
PyObject * allocMatrixWrapper(PyObject *self, PyObject *args){
	long unsigned int nbRow(0lu), nbCol(0lu);
	if(!PyArg_ParseTuple(args, "kk", &nbRow, &nbCol)){
		PyErr_SetString(PyExc_RuntimeError, "allocMatrixWrapper : wrong set of arguments. Expects two arguments for the matrix size\n");
		return NULL;
	}
	return allocMatrix(nbRow, nbCol);
}

