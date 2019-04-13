/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#ifndef __ASTERICS_ALLOC_H__
#define __ASTERICS_ALLOC_H__

void * asterics_malloc(long unsigned int sizeOfVectorInBytes);
void asterics_free(void* ptr);

long unsigned int getPitch(long unsigned int nbCol);
float * asterics_malloc2f(long unsigned int nbRow, long unsigned int nbCol);

#endif
