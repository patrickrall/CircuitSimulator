#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "stabilizer.h"

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

int isVectorWorking(gsl_vector *a, gsl_vector *b, int length){
	for(int i=0;i<length;i++){
		if(gsl_vector_get(a, i) - gsl_vector_get(b, i) > 0.0001){
			return 0;
		}
	}
	return 1;
}

int isMatrixWorking(gsl_matrix *a, gsl_matrix *b, int length){
	for(int i=0;i<length;i++){
		for(int j=0;j<length;j++){
			if(gsl_matrix_get(a, i, j) - gsl_matrix_get(b, i, j) > 0.0001){
				return 0;
			}
		}
	}
	return 1;
}

void testFileExponentialSum(){
	printf("\nTest file exponentialSum:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;

	fr = fopen("stabilizerTests/exponentialSumTests.txt", "rt");
	while(1){
		struct StabilizerStates state;
		int eps, outEps, p, outP, m, outM;
		gsl_complex ans;
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state.n)){
			break;
		}
		
		double vectorData[state.n];
		double matrixData[state.n*state.n];
		gsl_vector_view vectorView;
		gsl_matrix_view matrixView;
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state.k)){
			break;
		}
		
		//read Q
		if(!readInt(fr, line, maxLineLength, &state.Q)){
			break;
		}
		
		//read D
		if(!readArray(fr, line, maxLineLength, vectorData)){
			break;
		}
		vectorView = gsl_vector_view_array(vectorData, state.n);
		state.D = &vectorView.vector;
		
		//read J
		if(!readArray(fr, line, maxLineLength, matrixData)){
			break;
		}
		matrixView = gsl_matrix_view_array(matrixData, state.n, state.n);
		state.J = &matrixView.matrix;
		
		//read outEps
		if(!readInt(fr, line, maxLineLength, &outEps)){
			break;
		}
		
		//read outP
		if(!readInt(fr, line, maxLineLength, &outP)){
			break;
		}
		
		//read outM
		if(!readInt(fr, line, maxLineLength, &outM)){
			break;
		}
		
		exponentialSum(&state, &eps, &p, &m, &ans, 1);
		
		int isEpsWorking = eps == outEps ? 1 : 0;
		int isPWorking = p == outP ? 1 : 0;
		int isMWorking = mod(m, 8) == mod(outM, 8) ? 1 : 0;
		
		totalTests++;
		if(isEpsWorking*isPWorking*isMWorking > 0){
			successTests++;
		}
		else{
			printf("Test number %d failed.\n", totalTests);
		}
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
	
	printf("----------------------\n");
}

void testFileShrink(){
	printf("\nTest file shrink:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;
	
	fr = fopen("stabilizerTests/shrinkTests.txt", "rt");
	
	while(1){
		struct StabilizerStates state;
		int alpha, outk, outQ, outStatus;
		gsl_vector *xi, *outh, *outD;
		gsl_matrix *outG, *outGbar, *outJ;
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state.n)){
			break;
		}
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state.k)){
			break;
		}
		
		//read Q
		if(!readInt(fr, line, maxLineLength, &state.Q)){
			break;
		}
		
		//read alpha
		if(!readInt(fr, line, maxLineLength, &alpha)){
			break;
		}
		
		//read h
		double vectorDatah[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDatah)){
			break;
		}
		gsl_vector_view vectorViewh = gsl_vector_view_array(vectorDatah, state.n);
		state.h = &vectorViewh.vector;
		//printf("\nh: ");for(int i=0;i<state.n;i++){printf("%.0f ", gsl_vector_get(state.h, i));}
		
		//read D
		double vectorDataD[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataD)){
			break;
		}
		gsl_vector_view vectorViewD = gsl_vector_view_array(vectorDataD, state.n);
		state.D = &vectorViewD.vector;
		//printf("\nD: ");for(int i=0;i<state.n;i++){printf("%.0f ", gsl_vector_get(state.D, i));}
		
		//read xi
		double vectorDataxi[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataxi)){
			break;
		}
		gsl_vector_view vectorViewxi = gsl_vector_view_array(vectorDataxi, state.n);
		xi = &vectorViewxi.vector;
		
		//read G
		double matrixDataG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataG)){
			break;
		}
		gsl_matrix_view matrixViewG = gsl_matrix_view_array(matrixDataG, state.n, state.n);
		state.G = &matrixViewG.matrix;
		
		//read Gbar
		double matrixDataGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataGbar)){
			break;
		}
		gsl_matrix_view matrixViewGbar = gsl_matrix_view_array(matrixDataGbar, state.n, state.n);
		state.Gbar = &matrixViewGbar.matrix;
		
		//read J
		double matrixDataJ[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataJ)){
			break;
		}
		gsl_matrix_view matrixViewJ = gsl_matrix_view_array(matrixDataJ, state.n, state.n);
		state.J = &matrixViewJ.matrix;
		
		//read outStatus
		if(!readInt(fr, line, maxLineLength, &outStatus)){
			break;
		}
		
		//read outk
		if(!readInt(fr, line, maxLineLength, &outk)){
			break;
		}
		
		//read outQ
		if(!readInt(fr, line, maxLineLength, &outQ)){
			break;
		}
		
		//read outh
		double vectorDataouth[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataouth)){
			break;
		}
		gsl_vector_view vectorViewouth = gsl_vector_view_array(vectorDataouth, state.n);
		outh = &vectorViewouth.vector;
		
		//read outD
		double vectorDataoutD[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataoutD)){
			break;
		}
		gsl_vector_view vectorViewoutD = gsl_vector_view_array(vectorDataoutD, state.n);
		outD = &vectorViewoutD.vector;
		//printf("\noutD: ");for(int i=0;i<state.n;i++){printf("%.0f ", gsl_vector_get(outD, i));}
		
		//read outG
		double matrixDataoutG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutG)){
			break;
		}
		gsl_matrix_view matrixViewoutG = gsl_matrix_view_array(matrixDataoutG, state.n, state.n);
		outG = &matrixViewoutG.matrix;
		
		//read outGbar
		double matrixDataoutGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutGbar)){
			break;
		}
		gsl_matrix_view matrixViewoutGbar = gsl_matrix_view_array(matrixDataoutGbar, state.n, state.n);
		outGbar = &matrixViewoutGbar.matrix;
		
		//read outJ
		double matrixDataoutJ[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutJ)){
			break;
		}
		gsl_matrix_view matrixViewoutJ = gsl_matrix_view_array(matrixDataoutJ, state.n, state.n);
		outJ = &matrixViewoutJ.matrix;
		
		int status = shrink(&state, xi, alpha, 0);
		
		int isStatusWorking = status == outStatus ? 1 : 0;
		int iskWorking = state.k == outk ? 1 : 0;
		int isQWorking = state.Q == outQ ? 1 : 0;
		int ishWorking = isVectorWorking(state.h, outh, state.k);
		int isDWorking = isVectorWorking(state.D, outD, state.k);
		int isGWorking = isMatrixWorking(state.G, outG, state.k);
		int isGbarWorking = isMatrixWorking(state.Gbar, outGbar, state.k);
		int isJWorking = isMatrixWorking(state.J, outJ, state.k);
		
		//printf("%d %d %d %d %d %d %d %d ", isStatusWorking, iskWorking, isQWorking, ishWorking, isDWorking, isGWorking, isGbarWorking, isJWorking);
		
		totalTests++;
		if(isStatusWorking*iskWorking*isQWorking*ishWorking*isDWorking*isGWorking*isGbarWorking*isJWorking > 0){
			successTests++;
		}
		else{
			printf("Test number %d failed.\n", totalTests);
		}
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
	
	printf("----------------------\n");
}

void testFileInnerProduct(){
	printf("\nTest file inner product:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;
	
	fr = fopen("stabilizerTests/innerProductTests.txt", "rt");
	
	while(1){
		struct StabilizerStates state1;
		struct StabilizerStates state2;
		int eps, outEps, p, outP, m, outM;
		gsl_complex ans;
		
		//populate state1: 
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state1.n)){
			break;
		}
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state1.k)){
			break;
		}
		
		//read Q
		if(!readInt(fr, line, maxLineLength, &state1.Q)){
			break;
		}
		
		//read h
		double vectorDatah1[state1.n];
		if(!readArray(fr, line, maxLineLength, vectorDatah1)){
			break;
		}
		gsl_vector_view vectorViewh1 = gsl_vector_view_array(vectorDatah1, state1.n);
		state1.h = &vectorViewh1.vector;
		
		//read D
		double vectorDataD1[state1.n];
		if(!readArray(fr, line, maxLineLength, vectorDataD1)){
			break;
		}
		gsl_vector_view vectorViewD1 = gsl_vector_view_array(vectorDataD1, state1.n);
		state1.D = &vectorViewD1.vector;
		
		//read G
		double matrixDataG1[state1.n*state1.n];
		if(!readArray(fr, line, maxLineLength, matrixDataG1)){
			break;
		}
		gsl_matrix_view matrixViewG1 = gsl_matrix_view_array(matrixDataG1, state1.n, state1.n);
		state1.G = &matrixViewG1.matrix;
		
		//read Gbar
		double matrixDataGbar1[state1.n*state1.n];
		if(!readArray(fr, line, maxLineLength, matrixDataGbar1)){
			break;
		}
		gsl_matrix_view matrixViewGbar1 = gsl_matrix_view_array(matrixDataGbar1, state1.n, state1.n);
		state1.Gbar = &matrixViewGbar1.matrix;
		
		//read J
		double matrixDataJ1[state1.n*state1.n];
		if(!readArray(fr, line, maxLineLength, matrixDataJ1)){
			break;
		}
		gsl_matrix_view matrixViewJ1 = gsl_matrix_view_array(matrixDataJ1, state1.n, state1.n);
		state1.J = &matrixViewJ1.matrix;
		
		//populate state2: 
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state2.n)){
			break;
		}
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state2.k)){
			break;
		}
		
		//read Q
		if(!readInt(fr, line, maxLineLength, &state2.Q)){
			break;
		}
		
		//read h
		double vectorDatah2[state2.n];
		if(!readArray(fr, line, maxLineLength, vectorDatah2)){
			break;
		}
		gsl_vector_view vectorViewh2 = gsl_vector_view_array(vectorDatah2, state2.n);
		state2.h = &vectorViewh2.vector;
		
		//read D
		double vectorDataD2[state2.n];
		if(!readArray(fr, line, maxLineLength, vectorDataD2)){
			break;
		}
		gsl_vector_view vectorViewD2 = gsl_vector_view_array(vectorDataD2, state2.n);
		state2.D = &vectorViewD2.vector;
		
		//read G
		double matrixDataG2[state2.n*state2.n];
		if(!readArray(fr, line, maxLineLength, matrixDataG2)){
			break;
		}
		gsl_matrix_view matrixViewG2 = gsl_matrix_view_array(matrixDataG2, state2.n, state2.n);
		state2.G = &matrixViewG2.matrix;
		
		//read Gbar
		double matrixDataGbar2[state2.n*state2.n];
		if(!readArray(fr, line, maxLineLength, matrixDataGbar2)){
			break;
		}
		gsl_matrix_view matrixViewGbar2 = gsl_matrix_view_array(matrixDataGbar2, state2.n, state2.n);
		state2.Gbar = &matrixViewGbar2.matrix;
		
		//read J
		double matrixDataJ2[state2.n*state2.n];
		if(!readArray(fr, line, maxLineLength, matrixDataJ2)){
			break;
		}
		gsl_matrix_view matrixViewJ2 = gsl_matrix_view_array(matrixDataJ2, state2.n, state2.n);
		state2.J = &matrixViewJ2.matrix;
		
		//read outEps
		if(!readInt(fr, line, maxLineLength, &outEps)){
			break;
		}
		
		//read outP
		if(!readInt(fr, line, maxLineLength, &outP)){
			break;
		}
		
		//read outM
		if(!readInt(fr, line, maxLineLength, &outM)){
			break;
		}
		
		innerProduct(&state1, &state2, &eps, &p, &m, &ans, 1);
		
		int isEpsWorking = eps == outEps ? 1 : 0;
		int isPWorking = p == outP ? 1 : 0;
		int isMWorking = mod(m, 8) == mod(outM, 8) ? 1 : 0;
		
		totalTests++;
		if(eps==0&&isEpsWorking || isEpsWorking*isPWorking*isMWorking > 0){
			successTests++;
		}
		else{
			printf("eps: %d, outEps: %d, p: %d, outP: %d, m: %d, outM: %d", eps, outEps, p, outP, m, outM);
			printf("Test number %d failed.\n", totalTests);
		}
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
	
	printf("----------------------\n");
}

void testFileExtend(){
	printf("\nTest file extend:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;
	
	fr = fopen("stabilizerTests/extendTests.txt", "rt");
	
	while(1){
		struct StabilizerStates state;
		int outk;
		gsl_vector *xi;
		gsl_matrix *outG, *outGbar;
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state.n)){
			break;
		}
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state.k)){
			break;
		}
		
		//read xi
		double vectorDataxi[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataxi)){
			break;
		}
		gsl_vector_view vectorViewxi = gsl_vector_view_array(vectorDataxi, state.n);
		xi = &vectorViewxi.vector;
		
		//read G
		double matrixDataG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataG)){
			break;
		}
		gsl_matrix_view matrixViewG = gsl_matrix_view_array(matrixDataG, state.n, state.n);
		state.G = &matrixViewG.matrix;
		
		//read Gbar
		double matrixDataGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataGbar)){
			break;
		}
		gsl_matrix_view matrixViewGbar = gsl_matrix_view_array(matrixDataGbar, state.n, state.n);
		state.Gbar = &matrixViewGbar.matrix;
		
		//read outk
		if(!readInt(fr, line, maxLineLength, &outk)){
			break;
		}
		
		//read outG
		double matrixDataoutG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutG)){
			break;
		}
		gsl_matrix_view matrixViewoutG = gsl_matrix_view_array(matrixDataoutG, state.n, state.n);
		outG = &matrixViewoutG.matrix;
		
		//read outGbar
		double matrixDataoutGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutGbar)){
			break;
		}
		gsl_matrix_view matrixViewoutGbar = gsl_matrix_view_array(matrixDataoutGbar, state.n, state.n);
		outGbar = &matrixViewoutGbar.matrix;
		
		extend(&state, xi);
		
		int iskWorking = state.k == outk ? 1 : 0;
		int isGWorking = isMatrixWorking(state.G, outG, state.k);
		int isGbarWorking = isMatrixWorking(state.Gbar, outGbar, state.k);
		
		//printf("%d %d %d %d %d %d %d %d ", isStatusWorking, iskWorking, isQWorking, ishWorking, isDWorking, isGWorking, isGbarWorking, isJWorking);
		
		totalTests++;
		if(iskWorking*isGWorking*isGbarWorking > 0){
			successTests++;
		}
		else{
			printf("Test number %d failed.\n", totalTests);
		}
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
	
	printf("----------------------\n");
}

void testFileMeasurePauli(){
	printf("\nTest file measurePauli:\n");
	
	FILE *fr;
	const int maxLineLength = 20000;
	char line[maxLineLength];
	int totalTests = 0, successTests = 0;
	
	fr = fopen("stabilizerTests/measurePauliTests.txt", "rt");
	
	while(1){
		struct StabilizerStates state;
		int m, outk, outQ, outStatus;
		double outResult;
		gsl_vector *zeta, *xi, *outh, *outD;
		gsl_matrix *outG, *outGbar, *outJ;
		
		//read n
		if(!readInt(fr, line, maxLineLength, &state.n)){
			break;
		}
		
		//read k
		if(!readInt(fr, line, maxLineLength, &state.k)){
			break;
		}
		
		//read Q
		if(!readInt(fr, line, maxLineLength, &state.Q)){
			break;
		}
		
		//read m
		if(!readInt(fr, line, maxLineLength, &m)){
			break;
		}
		
		//read h
		double vectorDatah[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDatah)){
			break;
		}
		gsl_vector_view vectorViewh = gsl_vector_view_array(vectorDatah, state.n);
		state.h = &vectorViewh.vector;
		
		//read D
		double vectorDataD[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataD)){
			break;
		}
		gsl_vector_view vectorViewD = gsl_vector_view_array(vectorDataD, state.n);
		state.D = &vectorViewD.vector;
		
		//read zeta
		double vectorDatazeta[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDatazeta)){
			break;
		}
		gsl_vector_view vectorViewzeta = gsl_vector_view_array(vectorDatazeta, state.n);
		zeta = &vectorViewzeta.vector;
		
		//read xi
		double vectorDataxi[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataxi)){
			break;
		}
		gsl_vector_view vectorViewxi = gsl_vector_view_array(vectorDataxi, state.n);
		xi = &vectorViewxi.vector;
		
		//read G
		double matrixDataG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataG)){
			break;
		}
		gsl_matrix_view matrixViewG = gsl_matrix_view_array(matrixDataG, state.n, state.n);
		state.G = &matrixViewG.matrix;
		
		//read Gbar
		double matrixDataGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataGbar)){
			break;
		}
		gsl_matrix_view matrixViewGbar = gsl_matrix_view_array(matrixDataGbar, state.n, state.n);
		state.Gbar = &matrixViewGbar.matrix;
		
		//read J
		double matrixDataJ[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataJ)){
			break;
		}
		gsl_matrix_view matrixViewJ = gsl_matrix_view_array(matrixDataJ, state.n, state.n);
		state.J = &matrixViewJ.matrix;
		
		//read outResult
		if(!readDouble(fr, line, maxLineLength, &outResult)){
			break;
		}
		
		//read outk
		if(!readInt(fr, line, maxLineLength, &outk)){
			break;
		}
		
		//read outQ
		if(!readInt(fr, line, maxLineLength, &outQ)){
			break;
		}
		
		//read outh
		double vectorDataouth[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataouth)){
			break;
		}
		gsl_vector_view vectorViewouth = gsl_vector_view_array(vectorDataouth, state.n);
		outh = &vectorViewouth.vector;
		
		//read outD
		double vectorDataoutD[state.n];
		if(!readArray(fr, line, maxLineLength, vectorDataoutD)){
			break;
		}
		gsl_vector_view vectorViewoutD = gsl_vector_view_array(vectorDataoutD, state.n);
		outD = &vectorViewoutD.vector;
		
		//read outG
		double matrixDataoutG[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutG)){
			break;
		}
		gsl_matrix_view matrixViewoutG = gsl_matrix_view_array(matrixDataoutG, state.n, state.n);
		outG = &matrixViewoutG.matrix;
		
		//read outGbar
		double matrixDataoutGbar[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutGbar)){
			break;
		}
		gsl_matrix_view matrixViewoutGbar = gsl_matrix_view_array(matrixDataoutGbar, state.n, state.n);
		outGbar = &matrixViewoutGbar.matrix;
		
		//read outJ
		double matrixDataoutJ[state.n*state.n];
		if(!readArray(fr, line, maxLineLength, matrixDataoutJ)){
			break;
		}
		gsl_matrix_view matrixViewoutJ = gsl_matrix_view_array(matrixDataoutJ, state.n, state.n);
		outJ = &matrixViewoutJ.matrix;
		
		double result = measurePauli(&state, m, zeta, xi);
		
		int isResultWorking = result - outResult < 0.0001 ? 1 : 0;
		int iskWorking = state.k == outk ? 1 : 0;
		int isQWorking = state.Q == outQ ? 1 : 0;
		int ishWorking = isVectorWorking(state.h, outh, state.k);
		int isDWorking = isVectorWorking(state.D, outD, state.k);
		int isGWorking = isMatrixWorking(state.G, outG, state.k);
		int isGbarWorking = isMatrixWorking(state.Gbar, outGbar, state.k);
		int isJWorking = isMatrixWorking(state.J, outJ, state.k);
		
		totalTests++;
		if(isResultWorking*iskWorking*isQWorking*ishWorking*isDWorking*isGWorking*isGbarWorking*isJWorking > 0){
			successTests++;
		}
		else{
			printf("%d %d %d %d %d %d %d %d ", isResultWorking, iskWorking, isQWorking, ishWorking, isDWorking, isGWorking, isGbarWorking, isJWorking);
			printf("Test number %d failed.\n", totalTests);
		}
	}
	
	fclose(fr);
	
	printf("%d out of %d tests successful.\n", successTests, totalTests);
	
	printf("----------------------\n");
}

int main(){
	
	testFileExponentialSum();
	testFileShrink();
	testFileInnerProduct();
	testFileExtend();
	testFileMeasurePauli();
	
	return 0;
}
