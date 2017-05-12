#define _CUDA_COMPILER_

#include "support/common.h"
#include "support/partitioner.h"
#include <stdio.h>

#define BLOCK_SIZE 16
#define TILE_WIDTH_M 64
#define TILE_WIDTH_N 16
#define K (TILE_WIDTH_M/TILE_WIDTH_N)
#define TILE_WIDTH 16

#define BX 0
#define BY 0
#define BZ 0
#define TX 0
#define TY 0
#define TZ 0

//Original trash code
__global__ void gemm(float* A, float* B, float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        float* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += As[row][j] * Bs[j][col]; 
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue;
}

// __global__ void gemm(int n_tasks, float alpha, float* A, float* B, float* C, int m, int n, int k
//     #ifdef CUDA_8_0
//         , worklist
//     #endif
// ){

// 	int numARows = n;
// 	int numAColumns = m;
// 	int numBRows = n;
// 	int numBColumns = k;
// 	int numCRows = m;
// 	int numCColumns = k;


//   __shared__ float sm[K][TILE_WIDTH_N];

  
//   int tx = (threadIdx.x%TILE_WIDTH_N); 
//   int ty = (threadIdx.x/TILE_WIDTH_N);
//   int bx = blockIdx.x;  int by = blockIdx.y;
//   int Col = bx * TILE_WIDTH_M; 
//   int Row = by * TILE_WIDTH_N; 
  
//   float reg[K] = {0.0};
//   float temp[TILE_WIDTH_N] = {0.0};
  

//   for(int i=0; i<ceil(numBRows/(float)K); ++i)
//   {

//      for(int arridx = 0; arridx<K; ++arridx)
//      {
//         if(Col+threadIdx.x<numAColumns && (i*K+arridx) < numARows)
//           reg[arridx] = A[(i*K*numAColumns)+(numAColumns*arridx)+Col+threadIdx.x];
//         else
//           reg[arridx] = 0.0;
//      }
    
//      if(Row+tx<numBColumns && (i*K+ty) < numBRows  )
//         sm[ty][tx] = B[(i*K+ty)*numBColumns+ Row+tx];
//      else
//         sm[ty][tx] = 0.0;
//     __syncthreads();

//     for(int r1=0; r1 < K; ++r1)
//     {
//       for(int c1=0; c1<TILE_WIDTH_N; ++c1)
//         temp[c1] += reg[r1] * sm[r1][c1];
//     }
//       __syncthreads();
//   }
            
//   for(int c1=0; c1<TILE_WIDTH_N; ++c1)
//   {
//       if(Row+c1 < numCColumns&&Col+threadIdx.x <numCRows )
//         C[(Col+threadIdx.x)*numCColumns+Row+c1] = temp[c1];
//   }
// }

__global__ void transpose(float* A, float* B, int m, int n)
{
	__shared__ float sm[BLOCK_SIZE][BLOCK_SIZE];

	int tx = threadIdx.x; 	int ty = threadIdx.y;
	int bx = blockIdx.x; 	int by = blockIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;		

	if(row<m && col <n)
		sm[ty][tx] = A[row*n+col];
	__syncthreads();

	row = bx * blockDim.y + ty;
	col = by * blockDim.x + tx;

	if(row<n && col < m)
		B[row*m+col] = sm[tx][ty];
	__syncthreads();

	return;
}


// 32 single float array ->  32 bits unsigned int
__device__ unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) 
		b[i] = concatenate(&a[i*32]);
}

__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n)
{   

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n){
        float * array = new float[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}

// 32 bits unsigned int -> 32 single float array
// TODO: the array allocation should not be done here
__device__ float* deconcatenate(unsigned int x)
{
    float * array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size)
{ 
    float * array;
    
    for(int i=0; i<size; i+=32)
    {
        array = deconcatenate(a[i/32]);
        for (int k=0;k<32;k++) b[i+k] = array[k];
        delete[] array;
    }
}

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k) {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) 
		Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) 
		Csub[row*k+col] = -(2*(float)Cvalue-32*n);
}



void call_GPU_Kernel(int n_tasks, float alpha, int numAColumns, 
    int numARows, int numBColumns, int numBRows,
    int numCRows, int numCColumns, float *weights, float *x, float* output,
    float* xt,cudaStream_t stream)
{
    dim3 dimGrid0(ceil(numAColumns/(float)BLOCK_SIZE),
        ceil(numARows/(float)BLOCK_SIZE),1);
    dim3 dimBlock0(BLOCK_SIZE, BLOCK_SIZE, 1);
    transpose<<<dimGrid0, dimBlock0, 0, stream>>>(x,xt, numARows, numAColumns); 

    dim3 dimGrid(ceil(numARows/(float)TILE_WIDTH_M), ceil(numBColumns/(float)TILE_WIDTH_N), 1);
    dim3 dimBlock(TILE_WIDTH_M,1,1);
    gemm<<<dimGrid, dimBlock>>>(weights, output, numARows, numAColumns, numBColumns);

/*
    dim3 dimGrid(numBColumns/(float)BLOCK_SIZE, numARows/(float)BLOCK_SIZE,1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);
    gemm<<<dimGrid, dimBlock>>>(x, weights, output, numARows, numAColumns,
        numBColumns);
*/
}


// cudaError_t call_gemm_code(int blocks, int threads, int n_tasks, float alpha,
//     int in_size_i, int in_size_j, int out_size_i, int out_size_j,
//     int l_mem_size, XYZ* d_in, XYZ* d_out
// #ifdef CUDA_8_0
//     , int* worklist
// #endif
//     ){
//     dim3 dimGrid(blocks, 1);
//     dim3 dimBlock(threads, threads);
//     Bezier_surface<<<dimGrid, dimBlock, l_mem_size>>>(n_tasks, alpha, in_size_i, in_size_j, out_size_i, out_size_j,
//         d_in, d_out
// #ifdef CUDA_8_0
//         , worklist
// #endif
//         );
//     cudaError_t err = cudaGetLastError();
//     return err;
// }