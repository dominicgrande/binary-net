#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <chrono>
#define BLOCK_SIZE 16

// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
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
    if(i<size) b[i] = concatenate(&a[i*32]);
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
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n);
}

int main(){

    int n = 10000;
    float* A;
    A = new float[10000*4096];
    float* B;
    B = new float[4096*4096];

    for (int i=0; i<10000*4096; i++)
        A[i] = 1.0;

    for (int j=0; j<4096*4096; j++)
        B[j] = -1.0;

    float* A_Device; 
    float *B_Device; 
    unsigned int* A_Shrunk_Device; 
    unsigned int* B_Shrunk_Device;
    float*  C_Shrunk_Device;
    cudaMalloc(&A_Device, (size_t)10000*4096*sizeof(float));
    cudaMalloc(&B_Device, (size_t)4096*4096*sizeof(float));
    cudaMalloc(&A_Shrunk_Device, (size_t)(10000*4096*sizeof(unsigned int)/32));
    cudaMalloc(&B_Shrunk_Device, (size_t)(4096*4096*sizeof(unsigned int)/32));
    cudaMalloc(&C_Shrunk_Device, (size_t)(10000*4096*sizeof(float)));

    cudaMemcpy(A_Device, A, 10000*4096*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_Device, B, 4096*4096*sizeof(float), cudaMemcpyHostToDevice);

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    dim3 blockRows(64, 1, 1);
    dim3 gridRows(10000*4096/(64*32)+1, 1);
    concatenate_rows_kernel<<<gridRows, blockRows>>>(A_Device, A_Shrunk_Device, 10000*4096/32);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point end2= std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << std::endl;

    dim3 blockColumn(64, 1, 1);
    dim3 gridColumn(4096/64+1, 1);
    concatenate_cols_kernel<<<gridColumn, blockColumn>>>(B_Device, B_Shrunk_Device, 4096, 4096);

    dim3 xorBlock(16, 16, 1);

    dim3 xorGrid(4096/16+1, 10000/16+1);
    xnor_gemm<<<xorGrid, xorBlock>>>(A_Shrunk_Device, B_Shrunk_Device, C_Shrunk_Device, 4096, 10000/32, 4096);


    // float* C_Shrunk = new float[32*32];

    // cudaMemcpy(C_Shrunk, C_Shrunk_Device, 32*32*sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i=0; i<32*32; i++)
    // {
    //     C_Shrunk[i] = 0;
    // }

    // multiplyMatrices(B_Shrunk, A_Shrunk, C_Shrunk, 32, 2, 2, 32);
    // std::cout << "C values" << std::endl;
    // for (int x=0; x<32; x++)
    // {
    //     for (int y=0; y<32; y++)
    //         std::cout << std::setbase(16) << std::showbase <<(int)(C_Shrunk[x*32+y]) << " " ;

    //     std::cout << std::endl;
    // }
}

