#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <iomanip> //For string debugging -- Allows us to cast variables in different base
#include <iostream>
#include <string.h>
#include <cblas.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <chrono>
#include "xor.h"

using namespace std;
using namespace std::chrono;

int roundUp(int numToRound)
{

          int remainder = numToRound % 16;
              if (remainder == 0)
                        return numToRound;

                  return numToRound + 16 - remainder;
}

void matrixMulFunc(unsigned int *A, unsigned int *B, unsigned int *C, int numRowsStart, int numColsStart, int dimensions){
    int ib=16;
    int kb=16;
    int k=0;
    for (int ii = 0; ii <10000-numRowsStart; ii += 4*ib){ 
    for (int kk = 0; kk < 4096-numColsStart; kk += kb){
        for (int j=0; j < 128-numColsStart; j += 2){
            for(int i = ii; i < ii + ib; i += 2 ){
                if (i < dimensions-1 && j < dimensions-1 && k < dimensions-1){
		            int acc00, acc01, acc10, acc11;
                    if (kk == 0)
                        acc00 = acc01 = acc10 = acc11 = 0;
                    else {
                        acc00 = C[(128-numColsStart)*(i + 0)+j + 0];
                        acc01 = C[(128-numColsStart)*(i + 0)+j + 1];
                        acc10 = C[(128-numColsStart)*(i + 1) + j + 0];
                        acc11 = C[(128-numColsStart)*(i + 1) + j + 1];
                    }
                    for (k = kk; k < kk + kb; k++){ 
                            acc00 +=  __builtin_popcount((A[(128-numColsStart)*(i + 0)+k] ^ B[(128-numColsStart)*k+j + 0]));
                            acc01 +=  __builtin_popcount((A[(i + 0)*(128-numColsStart) + k] ^ B[k*(128-numColsStart) + j + 1]));
                            acc10 +=  __builtin_popcount((A[(i + 1)*(128-numColsStart)+ k] ^ B[k*(128-numColsStart) + j + 0]));
                            acc11 += __builtin_popcount((A[(i + 1)*(128-numColsStart )+ k] ^ B[k*(128-numColsStart)+j + 1]));
                    }
                C[(128-numColsStart)*(i + 0) + j + 0] = acc00;
                C[(128-numColsStart)*(i + 0) + j + 1] = acc01;
                C[(128-numColsStart)*(i + 1) + j + 0] = acc10;
                C[(128-numColsStart)*(i + 1) + j + 1] = acc11;
		        }
            }
        }
    }
}
}


void CPU_GPU_Xor(float * A, float * B, float * C, float alpha_1, float alpha_2, float alpha_3,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column,
                  float* A_device, float* B_device,
                  float* C_Device, unsigned int* Ac,
                  unsigned int* Bc, unsigned int*Ac_t,  unsigned int* aHostConcat2,
                  unsigned int* bHostConcat2){

    Timer        timer;
    cudaError_t  cudaStatus;

   

    const int A_GPU_Row = (int)alpha_1*A_Row;
    const int A_CPU_Row = A_Row - A_GPU_Row;

    // const int B_GPU_Col_End = roundUp((int)ceil(alpha_2*B_Column));
    // const int B_CPU_Col_Start = B_Column - B_GPU_Col_End;

    // float *A_device;
    // float *B_device;
    // float *C_Device;
     cudaStream_t kernel_stream;
     cudaStream_t data_stream;
     cudaStreamCreate(&data_stream); 
     cudaStreamCreate(&kernel_stream);

    // unsigned int *Ac;
    // unsigned int *Bc;

    int m = A_Row;
    int n = A_Column;
    int k = B_Column;

    int temp_A_Host_Size = sizeof(float)*A_CPU_Row*A_Column;
    float* temp_A_Host = (float *)malloc(temp_A_Host_Size);
    cudaMallocHost(&temp_A_Host,temp_A_Host_Size);

    // unsigned int* aHostConcat = new unsigned int[(A_CPU_Row_Start)*A_Column];
    // unsigned int* bHostConcat = new unsigned int[(B_Column*B_Row)/32];

    // cudaMalloc(&A_device, m*n*sizeof(float));
    // cudaMalloc(&B_device, n*k*sizeof(float));

    // cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_device, B, sizeof(float)*B_Column*B_Row, cudaMemcpyHostToDevice);

    // cudaMalloc(&Ac, (size_t)((m*n*sizeof(unsigned int))/32));
    // cudaMalloc(&Bc, (size_t)((n*k*sizeof(unsigned int))/32));
    // cudaMalloc(&C_Device, sizeof(float)*m*k);
   

     call_GPU_concatenate_rows(A_Column, A_Row, A_device, Ac, kernel_stream);
    // unsigned int* aHostConcat = new unsigned int[(A_CPU_Row_Start)*A_Column];
    //  unsigned int* bHostConcat = new unsigned int[(B_Column*B_Row)/32];

    
    // cudaStreamSynchronize(kernel_stream);
    // cudaMemcpyAsync(aHostConcat, &Ac[(A_Column*A_GPU_Row_End)/32], sizeof(unsigned int)*(A_CPU_Row_Start*n)/32, cudaMemcpyDeviceToHost, data_stream);

    call_GPU_concatenate_cols(A_Column, A_Row, B_Column, B_device, Bc, kernel_stream);
   
    
    cudaMemcpyAsync(temp_A_Host, &A[(A_GPU_Row)* A_Column], sizeof(float)*(int) (A_CPU_Row*A_Column), cudaMemcpyDeviceToHost, data_stream);
    cudaStreamSynchronize(kernel_stream);
     //cudaMemcpy(aHostConcat, &Ac[(A_Column*A_GPU_Row_End)/32], sizeof(unsigned int)*(A_CPU_Row_Start*n)/32, cudaMemcpyDeviceToHost);
    //  cudaMemcpyAsync(bHostConcat, Bc, sizeof(unsigned int)*(n*k)/32, cudaMemcpyDeviceToHost, data_stream);
     
    // cudaMemcpy(temp_A_Host, &A[(A_GPU_Row)* A_Column], sizeof(float)*(int) (A_CPU_Row*A_Column), cudaMemcpyDeviceToHost);

    call_GPU_xnor(A_Column, A_GPU_Row, B_Column, Ac, Ac_t, Bc, C_Device, kernel_stream);
    // int ib=16;

    // unsigned int* cHostConcat = new unsigned int[(A_CPU_Row_Start*k)/32]; 

    // int dimen = A_CPU_Row_Start*B_CPU_Col_Start;
    // std::thread first (matrixMulFunc, aHostConcat, bHostConcat, cHostConcat, A_CPU_Row_Start, 0, dimen);
    // std::thread second (matrixMulFunc, aHostConcat, bHostConcat, cHostConcat, A_CPU_Row_Start+ib, 0, dimen);
    // std::thread third (matrixMulFunc, aHostConcat, bHostConcat, cHostConcat, A_CPU_Row_Start+ib*2, 0, dimen);
    // std::thread fourth (matrixMulFunc, aHostConcat, bHostConcat, cHostConcat, A_CPU_Row_Start+ib*3, 0, dimen);
    // first.join();
    // second.join();
    // third.join();
    // fourth.join();
    if (alpha_1<1.0){


        // NONE STREAM

        // STREAM
        cudaStreamSynchronize (data_stream);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A_CPU_Row, B_Column, A_Column, 1, temp_A_Host, A_Column, B, B_Column, 0.0, C, B_Column);
        cudaMemcpyAsync(&C[A_GPU_Row * C_Column], C, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice, data_stream);
    }

    cudaDeviceSynchronize();

    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(data_stream);

    //std::cout << "First memcpy" << std::endl;
    //  std::cout << "The value of B_Column - B_CPU_Row_Start " << B_Column - B_CPU_Col_Start << std::endl;
    //  std::cout << "The value of B_Column is " << B_Column << std::endl;
    //  std::cout << "The value of B_CPU_Col_Start " << B_CPU_Col_Start << std::endl;
    //  std::cout << "The new value is " << (A_CPU_Row_Start*B_CPU_Col_Start)/32 << std::endl;
    //  std::cout << "The A value is " << A_CPU_Row_Start << std::endl;
    //  cudaMemcpy(&C_Device[128*A_GPU_Row_End], cHostConcat, sizeof(unsigned int)*((A_CPU_Row_Start*k))/32, cudaMemcpyHostToDevice);
    // // timer.start("Deallocation");
    // cudaMemcpy(C, C_Device, C_Column*C_Row*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //
    //std::cout << "Make it to after first memcpy" << std::endl;
    // std::cout << "Make it to after first memcpy" << std::endl;
    // unsigned int* cHost = new unsigned int[(C_Row*C_Column)/32];
    // cudaMemcpy(cHost, C_Device, (sizeof(float)*m*k)/32, cudaMemcpyHostToDevice);


    // std::cout << "The end value is: " << (C_Row*128)/32 << std::endl;

    // cudaFree(A_device);
    // cudaFree(B_device);
    // cudaFree(Ac);
    // cudaFree(Bc);
    // cudaFree(C_Device);
    // delete aHostConcat;
    // timer.stop("Allocation");
}

// Main ------------------------------------------------------------------------------------------
int main(){

    float* A;
    float* B;
    float* C;

    // float* A_device;
    // float* B_device;
    // float* C_device;

    int A_Row, A_Column, B_Row, B_Column, C_Row, C_Column;

    A_Row = 10000;
    A_Column = 4096;
    B_Column = 4096;
    B_Row = 4096;

    C_Row = A_Row;
    C_Column = B_Column;
    float alpha = 0.8;

    A = (float *)malloc(A_Row*A_Column*sizeof(float));
    B = (float *)malloc(B_Row*B_Column*sizeof(float));
    C = (float *)malloc(C_Row*C_Column*sizeof(float));

    // cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
    // cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
    // cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

    for (int i=0; i<A_Row*A_Column; i++)
        A[i] = 1.0;

    for (int i=0; i<B_Row*B_Column; i++)
        B[i] = 1.0;

    unsigned int *Bc_Host = new unsigned int[B_Row*B_Column/32];

    std::cout << "Right before CPU_GPU " << std::endl;

    float *A_device;
    float *B_device;
    float *C_Device;

    unsigned int *Ac;
    unsigned int *Ac_t;
    unsigned int *Bc;

    int m = A_Row;
    int n = A_Column;
    int k = B_Column;

    cudaMalloc(&A_device, m*n*sizeof(float));
    cudaMalloc(&B_device, n*k*sizeof(float));

    cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float)*B_Column*B_Row, cudaMemcpyHostToDevice);

    cudaMalloc(&Ac, (size_t)((m*n*sizeof(unsigned int))/32));
    cudaMalloc(&Ac_t, (size_t)((m*n*sizeof(unsigned int))/32));
    cudaMalloc(&Bc, (size_t)((n*k*sizeof(unsigned int))/32));
    cudaMalloc(&C_Device, sizeof(float)*m*k);
   
    cudaStream_t kernel_stream;
    cudaStream_t data_stream;
    cudaStreamCreate(&kernel_stream);
    cudaStreamCreate(&data_stream); 

    // call_GPU_concatenate_rows(A_Column, A_Row, A_device, Ac, kernel_stream);
    


//#    int timex [21];
//#    int iteration = 10;
//#    for(int j = 0; j<=20;  j += 1){
//#        alpha = .90 + j * .005;
//#        int timetemp = 0;
//#        for(int i = 0; i<iteration; i++){
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    CPU_GPU_Xor(A, B, C, alpha, alpha, alpha,
                            A_Row, A_Column,
                            B_Row, B_Column,
                            C_Row, C_Column,
                            A_device, B_device,
                            C_Device, Ac,
                            Bc,Ac_t, NULL,
                            NULL);

//                             std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
//
//            timetemp += (int)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
//
//        }
//        timex[j] = timetemp/iteration;
//        std::cout << "Alpha value " << alpha <<" time: " << timetemp/iteration << std::endl;
//        
//    }
//    // std::cout << " time: " << timetemp/iteration << std::endl;
//    for(auto const& value: timex)
//    {
//        std::cout << value << ",";
//    }


    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(Ac);
    cudaFree(Bc);
    cudaFree(C_Device);
    // delete aHostConcat;
    // delete bHostConcat;

    // cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_device, B, sizeof(float)*B_Row*B_Column, cudaMemcpyHostToDevice);

    // printf("After memcpy\n");

    // CPU_GPU_Gemm(A_device, B_device, C_device, alpha,
    //               A_Row, A_Column,
    //               B_Row, B_Column,
    //               C_Row, C_Column,
    //               B, C);

    // cudaMemcpy(C, C_device, sizeof(float)*C_Column*C_Row, cudaMemcpyDeviceToHost);

    // cudaFree(A_device);
    // cudaFree(B_device);
    // cudaFree(C_device);

//    for (int i=0; i<C_Column*10000; i++){
//        if( C[i] != 784){

//         std::cout << "WRONG: "<< "x: " <<i%4096 << " y: " << i/4096<< "  " << C[i] << std::endl;

//        }
//    }
    std::cout << "ALL GOOD" << std::endl;
    
    free(A);
    free(B);
    free(C);
    
	
// for (int i=0; i<C_Row; i++){
    //     for (int j=0; j<C_Column; j++){
    //         printf("%f ", C[i*C_Column+j]);
    //     }
    //     printf("\n");
    // }
}
