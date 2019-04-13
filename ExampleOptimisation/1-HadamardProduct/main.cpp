/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

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
	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabResult[i] = tabX[i]*tabY[i];
	}
}

///Get the number of cycles per elements of the Hadamard product
/**	@param nbElement : number of elements of the tables
 * 	@param nbRepetition : number of repetition to evaluate the function hadamard_product
*/
void evaluateHadamardProduct(long unsigned int nbElement, long unsigned int nbRepetition){
	//Allocation of the tables
	float * tabResult = new float[nbElement];
	float * tabX = new float[nbElement];
	float * tabY = new float[nbElement];
	//Initialisation of the tables
	for(long unsigned int i(0lu); i < nbElement; ++i){
		tabX[i] = (float)(i*32lu%17lu);
		tabY[i] = (float)(i*57lu%31lu);
	}
	//Stating the timer
	long unsigned int beginTime(rdtsc());
	for(long unsigned int i(0lu); i < nbRepetition; ++i){
		hadamard_product(tabResult, tabX, tabY, nbElement);
	}
	//Get the time of the nbRepetition calls
	long unsigned int elapsedTime((double)(rdtsc() - beginTime)/((double)nbRepetition));

	double cyclePerElement(((double)elapsedTime)/((double)nbElement));
	cout << "evaluateHadamardProduct : nbElement = "<<nbElement<<", cyclePerElement = " << cyclePerElement << " cy/el, elapsedTime = " << elapsedTime << " cy" << endl;
	cerr << nbElement << "\t" << cyclePerElement << "\t" << elapsedTime << endl;
	//Deallocate the tables
	delete[] tabResult;
	delete[] tabX;
	delete[] tabY;
}

int main(int argc, char** argv){
	cout << "Hadamard product" << endl;
	evaluateHadamardProduct(1000lu, 1000000lu);
	evaluateHadamardProduct(2000lu, 1000000lu);
	evaluateHadamardProduct(3000lu, 1000000lu);
	evaluateHadamardProduct(5000lu, 1000000lu);
	evaluateHadamardProduct(10000lu, 1000000lu);
	return 0;
}
