/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#ifndef __TIMER_H__
#define __TIMER_H__

#ifdef __i386
extern long unsigned int rdtsc(void);
#elif defined __amd64
extern long unsigned int rdtsc(void);
#endif

#endif

