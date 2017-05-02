#include "matrixMul.h"
#include "serialMatrixMul.h"
#include <stdio.h>

#define TILE_WIDTH 2

int main() {
  float *A; // The A matrix
  float *B; // The B matrix
  float *C; // The output C matrix

  float *A_device;
  float *B_device;
  float *C_device;

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this
                     
//Need to create input arrays
  numARows = 5;
  numAColumns = 5;
  numBRows = 5;
  numBColumns = 5;
  A = (float *) malloc(numARows*numAColumns*sizeof(float));
  B = (float *)malloc(numBRows*numBColumns*sizeof(float));
  
  for (int i=0; i<numARows; i++){
      for (int j=0; j<numAColumns; j++){
          A[i*numAColumns+j] = 5.0;
      }
  }

  for (int i=0; i<numBRows; i++){
      for (int j=0; j<numBColumns; j++){
          B[i*numBColumns+j] = 3.0;
      }
  }

  //@@ Set numCRows and numCColumns
  numCRows = numARows; 
  numCColumns = numBColumns;
  C = (float *)malloc(numCRows*numCColumns*sizeof(float));

  cudaMalloc(&A_device, numARows*numAColumns*sizeof(float));
  cudaMalloc(&B_device, numBRows*numBColumns*sizeof(float));
  cudaMalloc(&C_device, numCRows*numCColumns*sizeof(float));

  cudaMemcpy(A_device, A, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns/float(TILE_WIDTH)), ceil(numCRows/float(TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(A_device, B_device, C_device, numARows,
                                numAColumns, numBRows,
                                numBColumns, numCRows,
                                numCColumns);

  float * A_second;
  A_second = (float *)malloc(5*sizeof(float));
  for (int i=0; i<5; i++)
    printf("%f", A_second[i]);

  printf("\n");
  cudaMemcpy(A_second, &(A_device[20]), numAColumns*sizeof(float), cudaMemcpyDeviceToHost);
  serialMatrixMultiply(A_second, B, C, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, 4);

  cudaDeviceSynchronize();

  cudaMemcpy(C, C_device, 20*sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<numCRows; i++)
  {
      for (int j=0; j<numCColumns; j++){
          printf("%f ", C[i*numCColumns + j]);
      }
      printf("\n");
  }
  //@@ Copy the GPU memory back to the CPU here
  return 0;
}
