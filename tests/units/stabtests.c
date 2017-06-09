#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../libcirc/stabilizer/stabilizer.h"
// #include "../../libcirc/utils/matrix.h"

int readInt(FILE *fr, char *line, const int maxLineLength, int *target){
	if(!fgets(line, maxLineLength, fr)){
		return 0;
	}
	sscanf(line, "%d", target);
	return 1;
}

int readDouble(FILE *fr, char *line, const int maxLineLength, double *target){
	if(!fgets(line, maxLineLength, fr)){
		return 0;
	}
	sscanf(line, "%lf", target);
	return 1;
}

int readArray(FILE *fr, char *line, const int maxLineLength, double *target){
	char *currValue;
	int currIndex = 0;
	if(!fgets(line, maxLineLength, fr)){
		return 0;
	}
	currValue = strtok(line, ",");
	while(currValue != NULL) {
	  sscanf(currValue, "%lf", &target[currIndex++]);
	  currValue = strtok(NULL, ",");
	}
	return 1;
}

void testFileExponentialSum(){
	printf("\nTest file exponentialSum:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;

    time_t start_t, end_t;
    double diff_t;
    time(&start_t);
    
	fr = fopen("tests-c/exponentialSumTests.txt", "rt");

	while(1){
        int n, k;
		int eps, outEps, p, outP, m, outM;
		
		if(!readInt(fr, line, maxLineLength, &n)) break; //read n
		if(!readInt(fr, line, maxLineLength, &k)) break; //read k

		double vectorData[n];
		double matrixData[n*n];

	    struct StabilizerState* state = allocStabilizerState(n, k); 

		if(!readInt(fr, line, maxLineLength, &(state->Q))) break; //read Q
		
		//read D
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state->n; i++) setD(state, i, vectorData[i]);

		//read J
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->J, i, j, 1);
            }
        }
		
		if(!readInt(fr, line, maxLineLength, &outEps)) break; //read outEps
		if(!readInt(fr, line, maxLineLength, &outP)) break; //read outP
		if(!readInt(fr, line, maxLineLength, &outM)) break;	//read outM

		exponentialSumExact(state, &eps, &p, &m);
	
		int isEpsWorking = eps == outEps ? 1 : 0;
		int isPWorking = p == outP ? 1 : 0;
		int isMWorking = (m % 8) == (outM % 8) ? 1 : 0;
		
		totalTests++;
		if(isEpsWorking*isPWorking*isMWorking > 0) successTests++;
		else printf("Test number %d failed.\n", totalTests);

        freeStabilizerState(state);
	}
	
	fclose(fr);
	printf("%d out of %d tests successful.\n", successTests, totalTests);
    
    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Time elapsed: %f s\n", diff_t);

	printf("----------------------\n");
}

void testFileShrink(){
	printf("\nTest file shrink:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;

    time_t start_t, end_t;
    double diff_t;
    time(&start_t);

	fr = fopen("tests-c/shrinkTests.txt", "rt");
	
	while(1){
        int n, k;
		int alpha, outk, outQ, outStatus;
		
		if(!readInt(fr, line, maxLineLength, &n)) break; //read n
		if(!readInt(fr, line, maxLineLength, &k)) break; //read k

	    struct StabilizerState* state = allocStabilizerState(n, k); 

		if(!readInt(fr, line, maxLineLength, &state->Q)) break; //read Q
		if(!readInt(fr, line, maxLineLength, &alpha)) break; //read alpha
		
		double vectorData[n];
		double matrixData[n*n];

		//read h
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state->n; i++) BitVectorSet(state->h, i, vectorData[i]);
       
		//read D
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state->n; i++) setD(state, i, vectorData[i]);

		//read xi
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        struct BitVector *xi = newBitVector(n);
        for (int i = 0; i < state->n; i++) BitVectorSet(xi, i, vectorData[i]);
		
		//read G
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state->G);
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->G, i, j, 1);
            }
        }

		//read Gbar
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state->Gbar);
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->Gbar, i, j, 1);
            }
        }
		
		//read J
        if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->J, i, j, 1);
            }
        }

		if(!readInt(fr, line, maxLineLength, &outStatus)) break; //read outStatus
		if(!readInt(fr, line, maxLineLength, &outk)) break; //read outk
		if(!readInt(fr, line, maxLineLength, &outQ)) break; //read outQ
		
		//read outh
        if(!readArray(fr, line, maxLineLength, vectorData)) break;
        struct BitVector *outh = newBitVector(n);
        for (int i = 0; i < state->n; i++) BitVectorSet(outh, i, vectorData[i]);

		//read outD
		double outD[state->n];
		if(!readArray(fr, line, maxLineLength, outD)) break;
		
		//read outG
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        struct BitMatrix* outG = newBitMatrixZero(n,n);
        for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outG, i, j, 1);
            }
        }
		
		//read outGbar
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        struct BitMatrix* outGbar = newBitMatrixZero(n,n);
        for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outGbar, i, j, 1);
            }
        }
	
		//read outJ
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        struct BitMatrix* outJ = newBitMatrixZero(n,n);
        for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outJ, i, j, 1);
            }
        }

        int status = shrink(state, xi, alpha, 0);

		int isStatusWorking = status == outStatus ? 1 : 0;
		int iskWorking = state->k == outk ? 1 : 0;
		int isQWorking = state->Q == outQ ? 1 : 0;
		int ishWorking = BitVectorSame(state->h, outh);

		int isDWorking = 1;
        for (int i = 0; i < state->k; i++) {
            if (getD(state, i) != (int)outD[i]) {
                isDWorking = 0;
                break;
            }
        }

		int isGWorking = BitMatrixSame(state->G, outG);
		int isGbarWorking = BitMatrixSame(state->Gbar, outGbar);
		int isJWorking = BitMatrixSame(state->J, outJ);
      
		totalTests++;
		if(isStatusWorking*iskWorking*isQWorking*ishWorking*isDWorking*isGWorking*isGbarWorking*isJWorking > 0){
			successTests++;
		}
		else{
			printf("Test number %d failed.\n", totalTests);
		}
	
        freeStabilizerState(state);
        BitVectorFree(outh);
        BitMatrixFree(outG);
        BitMatrixFree(outGbar);
        BitMatrixFree(outJ);
    }
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Time elapsed: %f s\n", diff_t);

	printf("----------------------\n");
}


void testFileInnerProduct(){
	printf("\nTest file inner product:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;
	
    time_t start_t, end_t;
    double diff_t;
    time(&start_t);

	fr = fopen("tests-c/innerProductTests.txt", "rt");
	
	while(1){
		int eps, outEps, p, outP, m, outM;
        int n, k;
		
		//populate state1: 
		if(!readInt(fr, line, maxLineLength, &n)) break; //read n
		if(!readInt(fr, line, maxLineLength, &k)) break; //read k

	    struct StabilizerState* state1 = allocStabilizerState(n, k); 
		
		if(!readInt(fr, line, maxLineLength, &(state1->Q))) break; //read Q
	
		double vectorData[n];
		double matrixData[n*n];

		//read h
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state1->n; i++) BitVectorSet(state1->h, i, vectorData[i]);
       
		//read D
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state1->n; i++) setD(state1, i, vectorData[i]);

		//read G
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state1->G);
	    for (int i = 0; i < state1->n; i++) {
            for (int j = 0; j < state1->n; j++) {
                if (matrixData[i*state1->n + j] > 0) BitMatrixSet(state1->G, i, j, 1);
            }
        }

		//read Gbar
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state1->Gbar);
	    for (int i = 0; i < state1->n; i++) {
            for (int j = 0; j < state1->n; j++) {
                if (matrixData[i*state1->n + j] > 0) BitMatrixSet(state1->Gbar, i, j, 1);
            }
        }
		
		//read J
        if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state1->n; i++) {
            for (int j = 0; j < state1->n; j++) {
                if (matrixData[i*state1->n + j] > 0) BitMatrixSet(state1->J, i, j, 1);
            }
        }

		//populate state2: 
        if(!readInt(fr, line, maxLineLength, &n)) break; //read n
		if(!readInt(fr, line, maxLineLength, &k)) break; //read k
		
	    struct StabilizerState* state2 = allocStabilizerState(n, k); 
		
		if(!readInt(fr, line, maxLineLength, &(state2->Q))) break; //read Q
    
        if (n != state1->n) printf("STATES OF DIFFERENT SIZE\n");

		//read h
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state2->n; i++) BitVectorSet(state2->h, i, vectorData[i]);
       
		//read D
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state2->n; i++) setD(state2, i, vectorData[i]);
		
		//read G
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state2->G);
	    for (int i = 0; i < state2->n; i++) {
            for (int j = 0; j < state2->n; j++) {
                if (matrixData[i*state2->n + j] > 0) BitMatrixSet(state2->G, i, j, 1);
            }
        }

		//read Gbar
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state2->Gbar);
	    for (int i = 0; i < state2->n; i++) {
            for (int j = 0; j < state2->n; j++) {
                if (matrixData[i*state2->n + j] > 0) BitMatrixSet(state2->Gbar, i, j, 1);
            }
        }
		
		//read J
        if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state2->n; i++) {
            for (int j = 0; j < state2->n; j++) {
                if (matrixData[i*state2->n + j] > 0) BitMatrixSet(state2->J, i, j, 1);
            }
        }

		if(!readInt(fr, line, maxLineLength, &outEps)) break; //read outEps
		if(!readInt(fr, line, maxLineLength, &outP)) break; //read outP
		if(!readInt(fr, line, maxLineLength, &outM)) break; //read outM
	
		innerProductExact(state1, state2, &eps, &p, &m);
		
		int isEpsWorking = eps == outEps ? 1 : 0;
		int isPWorking = p == outP ? 1 : 0;
		int isMWorking = m % 8 == outM % 8 ? 1 : 0;
		
		totalTests++;
		if((eps==0&&isEpsWorking) || isEpsWorking*isPWorking*isMWorking > 0){
			successTests++;
		}
		else{
			printf("(%d, %d, %d) should be (%d, %d, %d)\n", eps, p, m, outEps, outP, outM);
			printf("Test number %d failed.\n", totalTests);
		}

        freeStabilizerState(state1);
        freeStabilizerState(state2);
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);

    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Time elapsed: %f s\n", diff_t);

	printf("----------------------\n");
}


void testFileMeasurePauli(){
	printf("\nTest file measurePauli:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;

    time_t start_t, end_t;
    double diff_t;
    time(&start_t);

	fr = fopen("tests-c/measurePauliTests.txt", "rt");
	
	while(1){
        int n, k;
		int m, outk, outQ;
		double outResult;
	
        if(!readInt(fr, line, maxLineLength, &n)) break; //read n
		if(!readInt(fr, line, maxLineLength, &k)) break; //read k

	    struct StabilizerState* state = allocStabilizerState(n, k); 

        if(!readInt(fr, line, maxLineLength, &(state->Q))) break; //read Q
		if(!readInt(fr, line, maxLineLength, &m)) break; //read m
		
		double vectorData[n];
		double matrixData[n*n];

		//read h
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state->n; i++) BitVectorSet(state->h, i, vectorData[i]);

        //read D
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < state->n; i++) setD(state, i, vectorData[i]);

        // read zeta
        struct BitVector* zeta = newBitVector(n);
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < n; i++) BitVectorSet(zeta, i, vectorData[i]);

        // read xi
        struct BitVector* xi = newBitVector(n);
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < n; i++) BitVectorSet(xi, i, vectorData[i]);
	
		// read G
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state->G);
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->G, i, j, 1);
            }
        }

		// read Gbar
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
        BitMatrixSetZero(state->Gbar);
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->Gbar, i, j, 1);
            }
        }
		
		// read J
        if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(state->J, i, j, 1);
            }
        }

		if(!readDouble(fr, line, maxLineLength, &outResult)) break; //read outResult
		
		if(!readInt(fr, line, maxLineLength, &outk)) break; //read outk
		
		if(!readInt(fr, line, maxLineLength, &outQ)) break; //read outQ
	
        // read xi
        struct BitVector* outh = newBitVector(n);
		if(!readArray(fr, line, maxLineLength, vectorData)) break;
        for (int i = 0; i < n; i++) BitVectorSet(outh, i, vectorData[i]);

		//read outD
		double outD[state->n];
		if(!readArray(fr, line, maxLineLength, outD)) break;
		
		//read outG
        struct BitMatrix* outG = newBitMatrixZero(n,n);
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outG, i, j, 1);
            }
        }
	
		//read outGbar
        struct BitMatrix* outGbar = newBitMatrixZero(n,n);
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outGbar, i, j, 1);
            }
        }
	
        //read outJ
        struct BitMatrix* outJ = newBitMatrixZero(n,n);
		if(!readArray(fr, line, maxLineLength, matrixData)) break;
	    for (int i = 0; i < state->n; i++) {
            for (int j = 0; j < state->n; j++) {
                if (matrixData[i*state->n + j] > 0) BitMatrixSet(outJ, i, j, 1);
            }
        }	

        if (totalTests < 11 && 0) {
            totalTests++;
            continue;
        }

		double result = measurePauli(state, m, zeta, xi);
		
		int isResultWorking = result - outResult < 0.0001 ? 1 : 0;
		int iskWorking = state->k == outk ? 1 : 0;
		int isQWorking = state->Q == outQ ? 1 : 0;
		int ishWorking = BitVectorSame(state->h, outh);

		int isDWorking = 1;
        for (int i = 0; i < state->k; i++) {
            if (getD(state, i) != (int)outD[i]) {
                isDWorking = 0;
                break;
            }
        }

        int isGWorking = BitMatrixSame(state->G, outG);
		int isGbarWorking = BitMatrixSame(state->Gbar, outGbar);

        int isJWorking = 1;
        for (int i = 0; i < state->k; i++) {
            for (int j = 0; j < state->k; j++) {
                if (BitMatrixGet(outJ, i, j) != BitMatrixGet(state->J, i, j)) {
                    isJWorking = 0;
                    break;
                }
            }
            if (!isJWorking) break;
        }

		totalTests++;

		if(isResultWorking*iskWorking*isQWorking*ishWorking*isDWorking*isGWorking*isGbarWorking*isJWorking > 0){
			successTests++;
		}
		else{
			printf("%d %d %d %d %d %d %d %d ", isResultWorking, iskWorking, isQWorking, ishWorking, isDWorking, isGWorking, isGbarWorking, isJWorking);
			printf("Test number %d failed.\n", totalTests);
		}

        freeStabilizerState(state);
        BitVectorFree(outh);
        BitVectorFree(xi);
        BitVectorFree(zeta);
        BitMatrixFree(outG);
        BitMatrixFree(outGbar);
        BitMatrixFree(outJ);
        //break;
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);

    time(&end_t);
    diff_t = difftime(end_t, start_t);
    printf("Time elapsed: %f s\n", diff_t);

	printf("----------------------\n");
}


int main(){

    for (int i = 0; i < 1; i++) {  
        testFileExponentialSum();
        testFileShrink();
        testFileInnerProduct();
        testFileMeasurePauli();
    }
	
	return 0;
}
