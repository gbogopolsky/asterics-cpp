#include <iostream>
#include "asterics_hpc.h"

using namespace std;

///Do the Hadamard product
/**	@param[out] ptabResult : table of results of tabX*tabY
 * 	@param scal : multiplication scalar (a)
 * 	@param ptabX : input table
 * 	@param ptabY : input table
 * 	@param nbElement : number of elements in the tables
*/
void saxpy(float* __restrict__ ptabResult, float scal, const float* __restrict__  ptabX, const float* __restrict__ ptabY, long unsigned int nbElement){
  const float* tabX = (const float*)__builtin_assume_aligned(ptabX, VECTOR_ALIGNEMENT);
  const float* tabY = (const float*)__builtin_assume_aligned(ptabY, VECTOR_ALIGNEMENT);
  float* tabResult = (float*)__builtin_assume_aligned(ptabResult, VECTOR_ALIGNEMENT);

	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabResult[i] = scal*tabX[i] + tabY[i];
	}
}

///Get the number of cycles per elements of the saxpy
/**	@param nbElement : number of elements of the tables
 * 	@param nbRepetition : number of repetition to evaluate the function saxpy
*/
void evaluateSaxpy(long unsigned int nbElement, long unsigned int nbRepetition){
	float * tabResult = (float*)asterics_malloc(sizeof(float)*nbElement);
	float * tabX = (float*)asterics_malloc(sizeof(float)*nbElement);
	float * tabY = (float*)asterics_malloc(sizeof(float)*nbElement);
	float scal(4.0f);
  
	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabX[i] = (float)(i*32lu%17lu);
		tabY[i] = (float)(i*57lu%31lu);
	}

	long unsigned int beginTime(rdtsc());
	for(long unsigned int i(0lu); i < nbRepetition; ++i){
		saxpy(tabResult, scal, tabX, tabY, nbElement);
	}
	long unsigned int elapsedTime((double)(rdtsc() - beginTime)/((double)nbRepetition));

	double cyclePerElement(((double)elapsedTime)/((double)nbElement));
	cout << "evaluateSaxpy : nbElement = "<<nbElement<<", cyclePerElement = " << cyclePerElement << " cy/el, elapsedTime = " << elapsedTime << " cy" << endl;
	cerr << nbElement << "\t" << cyclePerElement << "\t" << elapsedTime << endl;

	asterics_free(tabResult);
	asterics_free(tabX);
	asterics_free(tabY);
}

int main(int argc, char** argv){
	cout << "Saxpy" << endl;
	evaluateSaxpy(1000lu, 1000000lu);
	evaluateSaxpy(2000lu, 1000000lu);
	evaluateSaxpy(3000lu, 1000000lu);
	evaluateSaxpy(5000lu, 1000000lu);
	evaluateSaxpy(10000lu, 1000000lu);
	return 0;
}
