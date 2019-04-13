/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

//AVX intrinsic functions
#include <immintrin.h>

#include <iostream>
#include "asterics_hpc.h"

using namespace std;

///Do the Hadamard product
/**	@param[out] tabResult : table of results of tabX*tabY
 * 	@param tabX : input table
 * 	@param tabY : input table
 * 	@param nbElement : number of elements in the tables
*/
void hadamard_product(float* tabResult, const float* tabX, const float* tabY, long unsigned int nbElement){
	long unsigned int vecSize(VECTOR_ALIGNEMENT/sizeof(float));
	long unsigned int nbVec(nbElement/vecSize);
	for(long unsigned int i(0lu); i < nbVec; ++i){
		__m256 vecX = _mm256_load_ps(tabX + i*vecSize);
		__m256 vecY = _mm256_load_ps(tabY + i*vecSize);
		__m256 vecRes = _mm256_mul_ps(vecX, vecY);
		_mm256_store_ps(tabResult + i*vecSize, vecRes);
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

	double cyclePerElement(((double)elapsedTime)/((double)(nbElement)));
	cout << "evaluateHadamardProduct : nbElement = "<<nbElement<<", cyclePerElement = " << cyclePerElement << " cy/el, elapsedTime = " << elapsedTime << " cy" << endl;
	cerr << nbElement << "\t" << cyclePerElement << "\t" << elapsedTime << endl;
	asterics_free(tabResult);
	asterics_free(tabX);
	asterics_free(tabY);
}

int main(int argc, char** argv){
	cout << "Hadamard product intrinsics" << endl;
	evaluateHadamardProduct(1000lu, 5000000lu);
	evaluateHadamardProduct(1600lu, 5000000lu);
	evaluateHadamardProduct(2000lu, 5000000lu);
	evaluateHadamardProduct(2400lu, 5000000lu);
	evaluateHadamardProduct(2664lu, 3000000lu);
	evaluateHadamardProduct(3000lu, 3000000lu);
	evaluateHadamardProduct(4000lu, 3000000lu);
	evaluateHadamardProduct(5000lu, 1000000lu);
	evaluateHadamardProduct(10000lu, 1000000lu);
	return 0;
}
