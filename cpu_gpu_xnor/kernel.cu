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

// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)

__global__ void gemm(float* A, float* B, float* C, int m, int n, int k){

	int numARows = n;
	int numAColumns = m;
	int numBRows = n;
	int numBColumns = k;
	int numCRows = m;
	int numCColumns = k;


  __shared__ float sm[K][TILE_WIDTH_N];

  
  int tx = (threadIdx.x%TILE_WIDTH_N); 
  int ty = (threadIdx.x/TILE_WIDTH_N);
  int bx = blockIdx.x;  int by = blockIdx.y;
  int Col = bx * TILE_WIDTH_M; 
  int Row = by * TILE_WIDTH_N; 
  
  float reg[K] = {0.0};
  float temp[TILE_WIDTH_N] = {0.0};
  

  for(int i=0; i<ceil(numBRows/(float)K); ++i)
  {

     for(int arridx = 0; arridx<K; ++arridx)
     {
        if(Col+threadIdx.x<numAColumns && (i*K+arridx) < numARows)
          reg[arridx] = A[(i*K*numAColumns)+(numAColumns*arridx)+Col+threadIdx.x];
        else
          reg[arridx] = 0.0;
     }
    
     if(Row+tx<numBColumns && (i*K+ty) < numBRows  )
        sm[ty][tx] = B[(i*K+ty)*numBColumns+ Row+tx];
     else
        sm[ty][tx] = 0.0;
    __syncthreads();

    for(int r1=0; r1 < K; ++r1)
    {
      for(int c1=0; c1<TILE_WIDTH_N; ++c1)
        temp[c1] += reg[r1] * sm[r1][c1];
    }
      __syncthreads();
  }
            
  for(int c1=0; c1<TILE_WIDTH_N; ++c1)
  {
      if(Row+c1 < numCColumns&&Col+threadIdx.x <numCRows )
        C[(Col+threadIdx.x)*numCColumns+Row+c1] = temp[c1];
  }
}

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
__host__ __device__ unsigned int concatenate(float* array)
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



void call_GPU_concatenate_rows(int n, int m, float* A, unsigned int* Ac,
    cudaStream_t kernel_stream){

    int block_size = 64;

    // Concatenating the rows of A  

    // block = (block_size,1,1)
    // grid = (m*n/(block_size*32)+1,1)
    dim3 dimBlock(block_size,1,1);
    dim3 dimGrid(m*n/(block_size*32)+1, 1, 1);

    // concatenate_rows_kernel(A,Ac, np.intc(m*n/32), block= block, grid=grid)
    concatenate_rows_kernel<<<dimGrid, dimBlock, 0, kernel_stream>>>(A, Ac, m*n/32);

}
void call_GPU_concatenate_cols(int n, int m, int k, float* B, unsigned int* Bc,
    cudaStream_t kernel_stream){

    int block_size = 64;

    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(k/(block_size)+1, 1, 1);

    // concatenate_cols_kernel(B,Bc, np.intc(n), np.intc(k), block= block, grid=grid)
    concatenate_cols_kernel<<<dimGrid, dimBlock, 0, kernel_stream>>>(B, Bc, n, k);
}
void call_GPU_xnor(int n, int m, int k, unsigned int* Ac, unsigned int* Bc,
    float* C, cudaStream_t kernel_stream){


    int block_size = 16;
    // block = (block_size,block_size,1)
    // grid = (k / block_size + 1, m / block_size + 1) # better too many blocks than too little
    dim3 dimBlock_xnor(block_size, block_size, 1);
    dim3 dimGrid_xnor(k / block_size + 1, m / block_size + 1, 1);

    // xnor_kernel(Ac,Bc,C[0], np.intc(m), np.intc(n/32.), np.intc(k), block= block, grid=grid)
    xnor_gemm<<<dimGrid_xnor,dimBlock_xnor, 0, kernel_stream>>>(Ac, Bc, C, m, (int)n/32.0 , k); 

}

