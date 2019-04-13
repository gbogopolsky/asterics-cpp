/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#include <iostream>
#include "asterics_hpc.h"

using namespace std;

///Do the Hadamard product
/**	@param[out] ptabResult : table of results of tabX*tabY
 * 	@param ptabX : input table
 * 	@param ptabY : input table
 * 	@param nbElement : number of elements in the tables
*/
void hadamard_product(float* __restrict__ ptabResult, const float* __restrict__ ptabX, const float* __restrict__ ptabY, long unsigned int nbElement){
	const float* tabX = (const float*)__builtin_assume_aligned(ptabX, VECTOR_ALIGNEMENT);
	const float* tabY = (const float*)__builtin_assume_aligned(ptabY, VECTOR_ALIGNEMENT);
	float* tabResult = (float*)__builtin_assume_aligned(ptabResult, VECTOR_ALIGNEMENT);

	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabResult[i] = tabX[i]*tabY[i];
	}
}

///Get the number of cycles per elements of the Hadamard product
/**	@param nbElement : number of elements of the tables
 * 	@param nbRepetition : number of repetition to evaluate the function hadamard_product
*/
void evaluateHadamardProduct(long unsigned int nbElement, long unsigned int nbRepetition){
	float * tabResult = (float*)asterics_malloc(sizeof(float)*nbElement);
	float * tabX = (float*)asterics_malloc(sizeof(float)*nbElement);
	float * tabY = (float*)asterics_malloc(sizeof(float)*nbElement);

	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabX[i] = (float)(i*32lu%17lu);
		tabY[i] = (float)(i*57lu%31lu);
	}
	long unsigned int beginTime(rdtsc());
	for(long unsigned int i(0lu); i < nbRepetition; ++i){
		hadamard_product(tabResult, tabX, tabY, nbElement);
	}
	long unsigned int elapsedTime((double)(rdtsc() - beginTime)/((double)nbRepetition));

	double cyclePerElement(((double)elapsedTime)/((double)nbElement));
	cout << "evaluateHadamardProduct : nbElement = "<<nbElement<<", cyclePerElement = " << cyclePerElement << " cy/el, elapsedTime = " << elapsedTime << " cy" << endl;
	cerr << nbElement << "\t" << cyclePerElement << "\t" << elapsedTime << endl;
	asterics_free(tabResult);
	asterics_free(tabX);
	asterics_free(tabY);
}

int main(int argc, char** argv){
	cout << "Hadamard product vectorized" << endl;
	evaluateHadamardProduct(1000lu, 1000000lu);
	evaluateHadamardProduct(1500lu, 1000000lu);
	evaluateHadamardProduct(2000lu, 1000000lu);
	evaluateHadamardProduct(2500lu, 1000000lu);
	evaluateHadamardProduct(2666lu, 1000000lu);
	evaluateHadamardProduct(3000lu, 1000000lu);
	evaluateHadamardProduct(4000lu, 1000000lu);
	evaluateHadamardProduct(5000lu, 1000000lu);
	evaluateHadamardProduct(10000lu, 1000000lu);
	return 0;
}
