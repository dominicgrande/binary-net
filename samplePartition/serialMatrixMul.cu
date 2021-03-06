void serialMatrixMultiply(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int numAStart){

                                         //m is row first matrix, n is column first matrix
                                         //p is row second matrix, q is column second matrix
    for(int i=numAStart; i<numARows; ++i){
        for(int j=0; j<numBColumns; ++j){
            C[i][j]=0;
            for(int k=0; k<numAColumns; ++k)
                C[i][j]=C[i][j]+(A[i][k]*B[k][j]);
        }
    }
}

