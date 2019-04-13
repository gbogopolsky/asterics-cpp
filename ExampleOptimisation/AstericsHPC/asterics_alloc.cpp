/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#ifndef __APPLE__
#	include <malloc.h>
#else
#	include <stdlib.h>
#endif

#include <string.h>
#include "asterics_alloc.h"

#ifdef __APPLE__
///Alloc an aligned vector
/**	@param sizeOfVectorInBytes : size of the vector xe want to allocate
 * 	@param alignementInBytes : alignement of the vector we want to allocate
 * 	@return aligned pointor of the vector
*/
void * memalign(long unsigned int alignementInBytes, long unsigned int sizeOfVectorInBytes){
	void * ptr = NULL;
	posix_memalign(&ptr, alignementInBytes, sizeOfVectorInBytes);
	return ptr;
}
#endif

///Get the pitch of a matrix
/**	@param nbCol : number of columns of the matrix
 * 	@return pitch of the matrix
*/
long unsigned int getPitch(long unsigned int nbCol){
	long unsigned int vecSize(VECTOR_ALIGNEMENT/sizeof(float));
	long unsigned int pitch(vecSize - (nbCol % vecSize));
	if(pitch == vecSize){pitch = 0lu;}
	return pitch;
}

///Do the aligned allocation of a pointer
/**	@param sizeOfVectorInBytes : number of bytes to be allocated
 * 	@return allocated pointer
*/
void * asterics_malloc(long unsigned int sizeOfVectorInBytes){
	return memalign(VECTOR_ALIGNEMENT, sizeOfVectorInBytes + getPitch(sizeOfVectorInBytes));
}

///Free an aligned pointer
/**	@param ptr : ptr to be freed
*/
void asterics_free(void* ptr){
	free(ptr);
}

///Allocate a 2d matrix with a pitch for float
/**	@param nbRow : number of rows of the matrix
 * 	@param nbCol : number of columns of the matrix
 * 	@return allocated matrix
*/
float * asterics_malloc2f(long unsigned int nbRow, long unsigned int nbCol){
	long unsigned int pitch(getPitch(nbCol));
	long unsigned int sizeByte(sizeof(float)*nbRow*(nbCol + pitch));
	float* mat = (float*)asterics_malloc(sizeByte);
	memset(mat, 0, sizeByte);			//Do not forget to initialse the values of the matrix which are in the pitch
	return mat;
}

