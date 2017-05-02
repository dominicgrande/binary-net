#include "matrixMul.h"
#include "serialMatrixMul.h"
#include <stdio.h>

#define TILE_WIDTH 64

int main() {
  float *A; // The A matrix
  float *B; // The B matrix
  float *C; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this
                     
//Need to create input arrays
  numARows = 5;
  numAColumns = 5;
  numBRows = 5;
  numBColumns = 5;
  cudaMallocManaged(&A, numARows*numAColumns*sizeof(float));
  cudaMallocManaged(&B, numBRows*numBColumns*sizeof(float));
  
  for (int i=0; i<numARows; i++){
      for (int j=0; j<numAColumns; j++){
          A[i*numAColumns+j] = 5.0;
      }
  }

  for (int i=0; i<numBRows; i++){
      for (int j=0; j<numBColumns; j++){
          B[i*numAColumns+j] = 3.0;
      }
  }

  //@@ Set numCRows and numCColumns
  numCRows = numARows; 
  numCColumns = numBColumns;
  cudaMallocManaged(&C, numCRows*numCColumns*sizeof(float));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns/float(TILE_WIDTH)), ceil(numCRows/float(TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(A, B, C, numARows-1,
                                numAColumns, numBRows,
                                numBColumns, numCRows,
                                numCColumns);


  serialMatrixMultiply(A, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, 5);

  for (int i=0; i<numCRows; i++)
  {
      for (int j=0; j<numCColumns; j++){
          printf("%f ", C[i*numCColumns + j]);
      }
      printf("\n");
  }

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  

  return 0;
}
