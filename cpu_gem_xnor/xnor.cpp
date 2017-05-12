// #include "support/cuda-setup.h"
#include "kernel.h"
// #include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <iostream>
#include <string.h>
#include <cblas.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <chrono>


// using namespace std;
// using namespace std::chrono;

static float* B_Host;

static float* A_Host;
static int* At;
static unsigned int* Ac;
static unsigned int* Bc;
static float* C_Host;
static int pinned_memory;

// int roundUp(int numToRound)
// {

//           int remainder = numToRound % 16;
//               if (remainder == 0)
//                         return numToRound;

//                   return numToRound + 16 - remainder;
// }

// void matrixMulFunc(unsigned int *A, unsigned int *B, unsigned int *C, int numRowsStart, int numColsStart, int dimensions){
//     int ib=16;
//     int kb=16;
//     int k=0;
//     for (int ii = 0; ii <10000-numRowsStart; ii += 4*ib){ 
//     for (int kk = 0; kk < 4096-numColsStart; kk += kb){
//         for (int j=0; j < 128-numColsStart; j += 2){
//             for(int i = ii; i < ii + ib; i += 2 ){
//                 if (i < dimensions-1 && j < dimensions-1 && k < dimensions-1){
// 		            int acc00, acc01, acc10, acc11;
// //                  std::cout << "Current i value " << i << ". The Current K value is " << k << std::endl;                
//                     if (kk == 0)
//                         acc00 = acc01 = acc10 = acc11 = 0;
//                     else {
//                         acc00 = C[(128-numColsStart)*(i + 0)+j + 0];
//                         acc01 = C[(128-numColsStart)*(i + 0)+j + 1];
//                         acc10 = C[(128-numColsStart)*(i + 1) + j + 0];
//                         acc11 = C[(128-numColsStart)*(i + 1) + j + 1];
//                     }
//                     for (k = kk; k < kk + kb; k++){ 
//                             acc00 +=  __builtin_popcount((A[(128-numColsStart)*(i + 0)+k] ^ B[(128-numColsStart)*k+j + 0]));
//                             acc01 +=  __builtin_popcount((A[(i + 0)*(128-numColsStart) + k] ^ B[k*(128-numColsStart) + j + 1]));
//                             acc10 +=  __builtin_popcount((A[(i + 1)*(128-numColsStart)+ k] ^ B[k*(128-numColsStart) + j + 0]));
//                             acc11 += __builtin_popcount((A[(i + 1)*(128-numColsStart )+ k] ^ B[k*(128-numColsStart)+j + 1]));
//                     }
//                 C[(128-numColsStart)*(i + 0) + j + 0] = acc00;
//                 C[(128-numColsStart)*(i + 0) + j + 1] = acc01;
//                 C[(128-numColsStart)*(i + 1) + j + 0] = acc10;
//                 C[(128-numColsStart)*(i + 1) + j + 1] = acc11;
// 		        }
//             }
//         }
//     }
// }
// }


// void CPU_GPU_Xor(float * A, float * B, float * C, float alpha
//                   int A_Row, int A_Column,
//                   int B_Row, int B_Column,
//                   int C_Row, int C_Column,
//                   float* A_device, float* B_device,
//                   float* C_Device, unsigned int* Ac,
//                   unsigned int* Bc, unsigned int* aHostConcat2,
//                   unsigned int* bHostConcat2){

// void CPU_GPU_Xnor( * A, float * B, float * C, float alpha,
void CPU_GPU_Xnor(unsigned long A_ptr, unsigned long B_ptr, unsigned long C_ptr, float alpha,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column){

    Timer        timer;
    cudaError_t  cudaStatus;

    float * A =  (float*)A_ptr;
    float * B =  (float*)B_ptr;
    float * C =  (float*)C_ptr;

    cudaStream_t kernel_stream;
    cudaStream_t data_stream;
    cudaStreamCreate(&kernel_stream);
    cudaStreamCreate(&data_stream);

    const int A_GPU_Row     = (int)( A_Row * alpha);
    const int A_CPU_Row     = A_Row - A_GPU_Row;
    int temp_A_Host_Size = sizeof(float)*A_CPU_Row*A_Column;


    call_GPU_concatenate_rows(A_Column, A_Row, A, Ac, kernel_stream);
    call_GPU_concatenate_cols(A_Column, A_Row, B_Column, B, Bc, kernel_stream);

    cudaStreamSynchronize(kernel_stream);

    cudaMemcpyAsync(A_Host,&A[(A_GPU_Row)* A_Column], temp_A_Host_Size ,cudaMemcpyDeviceToHost, data_stream);
    call_GPU_xnor(A_Column, A_Row, B_Column, Ac, Bc, C, kernel_stream);

    if (alpha<1.0){

        cudaStreamSynchronize (data_stream);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A_CPU_Row, B_Column, A_Column, 1, A_Host, A_Column, B_Host, B_Column, 0.0, C_Host, B_Column);
        cudaMemcpyAsync(&C[A_GPU_Row * C_Column], C_Host, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice, data_stream);
    }

    cudaDeviceSynchronize();
    
    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(data_stream);

    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(data_stream);



}

// void Load_Weights(pybind11::array_t<float> vec)
// {
//     // py::buffer_info info_i = i.request();
//     auto info_i = vec.request();

//     // cudaError_t error = cudaMalloc(&w, info_i.size * sizeof(float));
// //   cudaError_t error = cudaMalloc(&gpu_ptr, size * sizeof(double));

//     B_Host = (float*)info_i.ptr;
//     printf("Host value");
//     for(int i = 0; i < 50 ; i++){
//       printf("%f,",B_Host[i]);
//     }
// }
// void initMemory(float alpha, int pinned_memory_input,
//                 int A_Row, int A_Column,
//                 int B_Row, int B_Column,
//                 int C_Row, int C_Column)
// {
//     // A_host = (float *)malloc(A_Row*A_Column*sizeof(float));
//     // C_host = (float *)malloc(C_Row*C_Column*sizeof(float));
//     pinned_memory = pinned_memory_input;
//     int height = A_Row -  (int)( A_Row * alpha);

//     if(pinned_memory == 1){
//         cudaMallocHost(&A_Host,height * A_Column);
//         cudaMallocHost(&C_Host,height * C_Column);

//     }
//     else{
//         A_Host = new float [height*A_Column];
//         C_Host = new float [height*C_Column];
//     }
//     cudaMalloc(&At, A_Row*A_Column*sizeof(float));
//     cudaMalloc(&Ac,A_Row * A_Column/32 * sizeof(unsigned int));
//     cudaMalloc(&Bc,B_Row * B_Column/32 * sizeof(unsigned int));


// }

// void deallocateMemory(int alpha,
//                 int A_Row, int A_Column,
//                 int B_Row, int B_Column,
//                 int C_Row, int C_Column)
// {
//     // A_host = (float *)malloc(A_Row*A_Column*sizeof(float));
//     // C_host = (float *)malloc(C_Row*C_Column*sizeof(float));

//     if(pinned_memory == 1){
//         cudaFreeHost(A_Host);
//         cudaFreeHost(C_Host);
//     }
//     else{
//         delete [] A_Host;
//         delete [] C_Host;
//     }
//     cudaFree(At);
//     cudaFree(Ac);
//     cudaFree(Bc);
  
// }
// Main ------------------------------------------------------------------------------------------
// int main(){

//     float* A;
//     float* B;
//     float* C;

//     // float* A_device;
//     // float* B_device;
//     // float* C_device;

//     int A_Row, A_Column, B_Row, B_Column, C_Row, C_Column;

//     A_Row = 10000;
//     A_Column = 4096;
//     B_Column = 4096;
//     B_Row = 4096;

//     C_Row = A_Row;
//     C_Column = B_Column;
//     float alpha = 0.8;

//     A = (float *)malloc(A_Row*A_Column*sizeof(float));
//     B = (float *)malloc(B_Row*B_Column*sizeof(float));
//     C = (float *)malloc(C_Row*C_Column*sizeof(float));

//     // cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
//     // cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
//     // cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

//     for (int i=0; i<A_Row*A_Column; i++)
//         A[i] = 1.0;

//     for (int i=0; i<B_Row*B_Column; i++)
//         B[i] = 1.0;

//     unsigned int *Bc_Host = new unsigned int[B_Row*B_Column/32];

//     std::cout << "Right before CPU_GPU " << std::endl;

//     float *A_device;
//     float *B_device;
//     float *C_Device;

//     unsigned int *Ac;
//     unsigned int *Bc;

//     int m = A_Row;
//     int n = A_Column;
//     int k = B_Column;

//     cudaMalloc(&A_device, m*n*sizeof(float));
//     cudaMalloc(&B_device, n*k*sizeof(float));

//     cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
//     cudaMemcpy(B_device, B, sizeof(float)*B_Column*B_Row, cudaMemcpyHostToDevice);

//     cudaMalloc(&Ac, (size_t)((m*n*sizeof(unsigned int))/32));
//     cudaMalloc(&Bc, (size_t)((n*k*sizeof(unsigned int))/32));
//     cudaMalloc(&C_Device, sizeof(float)*m*k);
   
//     cudaStream_t kernel_stream;
//     cudaStream_t data_stream;
//     cudaStreamCreate(&kernel_stream);
//     cudaStreamCreate(&data_stream); 

//     // call_GPU_concatenate_rows(A_Column, A_Row, A_device, Ac, kernel_stream);
    
//     std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//     std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
//     std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
//     return 0;


//     int timex [21];
//     int iteration = 1;
//     for(int j = 0; j<=20;  j += 1){
//         alpha = j * .05;
//         int timetemp = 0;
//         for(int i = 0; i<iteration; i++){
//             std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//                 CPU_GPU_Xor(A_device, B_device, C_Device, alpha, 
//                             A_Row, A_Column,
//                             B_Row, B_Column,
//                             C_Row, C_Column);

//             std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

//             timetemp += (int)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

//         }
//         timex[j] = timetemp/iteration;
//         std::cout << "Alpha value " << alpha <<" time: " << timetemp/iteration << std::endl;
        
//     }
//     // std::cout << " time: " << timetemp/iteration << std::endl;
//     for(auto const& value: timex)
//     {
//         std::cout << value << ",";
//     }


//     cudaFree(A_device);
//     cudaFree(B_device);
//     cudaFree(Ac);
//     cudaFree(Bc);
//     cudaFree(C_Device);
//     // delete aHostConcat;
//     // delete bHostConcat;

//     // cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
//     // cudaMemcpy(B_device, B, sizeof(float)*B_Row*B_Column, cudaMemcpyHostToDevice);

//     // printf("After memcpy\n");

//     // CPU_GPU_Gemm(A_device, B_device, C_device, alpha,
//     //               A_Row, A_Column,
//     //               B_Row, B_Column,
//     //               C_Row, C_Column,
//     //               B, C);

//     // cudaMemcpy(C, C_device, sizeof(float)*C_Column*C_Row, cudaMemcpyDeviceToHost);

//     // cudaFree(A_device);
//     // cudaFree(B_device);
//     // cudaFree(C_device);

// //    for (int i=0; i<C_Column*10000; i++){
// //        if( C[i] != 784){

// //         std::cout << "WRONG: "<< "x: " <<i%4096 << " y: " << i/4096<< "  " << C[i] << std::endl;

// //        }
// //    }
//     std::cout << "ALL GOOD" << std::endl;
    
//     free(A);
//     free(B);
//     free(C);
    
	
// }


PYBIND11_PLUGIN(xnor)
{
  pybind11::module m("xnor", "GPU Library");
  m.def("CPU_GPU_Xnor", CPU_GPU_Xnor);
  
}
//   m.def("Load_Weights", Load_Weights);
//   m.def("initMemory", initMemory);
//   m.def("deallocateMemory", deallocateMemory);