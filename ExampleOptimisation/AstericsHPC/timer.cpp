/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/


#include "timer.h"

#ifdef __i386
///Get the number of cycles since the begining of the program
/**	@return number of cycles since the begining of the program
*/
extern long unsigned int rdtsc(void) {
	long unsigned int x;
	__asm__ volatile ("rdtsc" : "=A" (x));
	return x;
}
#elif defined __amd64
///Get the number of cycles since the begining of the program
/**	@return number of cycles since the begining of the program
*/
extern long unsigned int rdtsc(void) {
	long unsigned int a, d;
	__asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
	return (d<<32) | a;
}
#endif

