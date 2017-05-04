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
#include "xor.h"


void CPU_GPU_Xor(float * A, float * B, float * C, float alpha_1, float alpha_2, float alpha_3,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column,
                  unsigned int* B_Host, float* C_Host){

    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    unsigned int *Ac;
    unsigned int *Bc;

    // const int A_GPU_Row     = (int) A_Row * alpha;
    // const int A_CPU_Row     = A_Row - A_GPU_Row;

    const int A_GPU_Row_End = (int)alpha_1*A_Row;
    const int A_CPU_Row_Start = A_Row - A_GPU_Row_End;

    const int B_GPU_Col_End = (int)alpha_2*B_Column;
    const int B_CPU_Col_Start = B_Column - B_GPU_Col_End;

    //Need to implement alpha value for computation

    float *A_device;
    float *B_device;
    float *C_Device;

    
    int m = A_Row;
    int n = A_Column;
    int k = B_Column;

    cudaMalloc(&A_device, A_GPU_Row_End*n*sizeof(float));
    cudaMalloc(&B_device, n*B_GPU_Col_End*sizeof(float));

    cudaMemcpy(A_device, A, sizeof(float)*A_GPU_Row_End*A_Column, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float)*B_Column*B_Row, cudaMemcpyHostToDevice);

    cudaMalloc(&Ac, (size_t)((m*n*sizeof(unsigned int))/32));
    cudaMalloc(&Bc, (size_t)((n*k*sizeof(unsigned int))/32));
    cudaMalloc(&C_Device, sizeof(float)*m*k);
   
  


    // timer.start("Kernel Call");

    //n, m, A, A_c
    // timer.start("Concat");
    call_GPU_concatenate_rows(A_Column, A_GPU_Row_End, A_device, Ac);
    unsigned int* aHostConcat = new unsigned int[A_Column*(A_Row-A_CPU_Row_Start)];
    concatenate_rows_serial(&A[A_Column*A_CPU_Row_Start], aHostConcat, 
                                A_Row-A_CPU_Row_Start, A_Column);

    cudaMemcpy(&Ac[A_Column*A_CPU_Row_Start], aHostConcat, A_Column*(A_Row-A_CPU_Row_Start)*sizeof(unsigned int),
                cudaMemcpyHostToDevice);
    
    call_GPU_concatenate_cols(A_Column, A_Row, B_Column, B_device, Bc);
    cudaDeviceSynchronize();
    // // timer.stop("Concat");
    // // timer.print("Concat", 1);
    

      timer.start("B");
     cudaMemcpy(B_Host, B_device, sizeof(unsigned int)*B_Column*B_Row/32, cudaMemcpyDeviceToHost);
     timer.stop("B");
     timer.print("B", 1);
    // // unsigned int* bHostConcat = new unsigned int[B_Row*(B_Column-B_CPU_Col_Start)];
    // // concatenate_cols_serial(B, bHostConcat, B_Row, B_CPU_Col_Start);
    // // cudaMemcpy(&Bc[A_row], bHostConcat, B_Row*(B_Column-B_CPU_Col_Start)*sizeof(unsigned int), cudaMemcpyHostToDevice);
                            
    // // timer.start("Kernel");
     call_GPU_xnor(A_Column, A_Row, B_Column, Ac, Bc, C_Device);
     cudaDeviceSynchronize();
    // // timer.stop("Kernel");
    // // timer.print("Kernel", 1);

    // // timer.start("Deallocation");
    // cudaMemcpy(C, C_Device, C_Column*C_Row*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(Ac);
    cudaFree(Bc);
    cudaFree(C_Device);
    delete aHostConcat;
    timer.stop("Allocation");
    timer.print("Allocation", 1);
    

    // void call_GPU_concatenate_rows(int n, int m, float* A, float* Ac);
    // void call_GPU_concatenate_cols(int n, int m, int k, float* B, float* Bc);
    // void call_GPU_xnor(int n, int m, int k, float* Ac, float* Bc, float* C);
    //Changed the A_GPU_Row start with altered alpha value

    // void call_GPU_concatenate_rows(A_Column, A_Row, A, Ac);
    // printf("Made it after GPU kernel. Need sync\n");
    // float* temp_A_Host;
    // if (alpha<1){
    //     temp_A_Host = (float *)malloc(sizeof(float)*A_CPU_Row*A_Column);

    //     cudaMemcpy(temp_A_Host, &A[(A_GPU_Row)* A_Column], sizeof(float)*(int) (A_CPU_Row*A_Column), cudaMemcpyDeviceToHost);

    //     printf("Memcpy is no good.\n");

    //     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
    //             A_CPU_Row, B_Column, A_Column, 1, temp_A_Host, A_Column, B_Host, B_Column, 0.0, C_Host, B_Column);
    //     free(temp_A_Host);
    // }

    // cudaDeviceSynchronize();
    // cudaMemcpy(&C[A_GPU_Row * C_Column], C_Host, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice);
    // timer.stop("Kernel Call");
    // timer.print("Kernel Call", 1);
    // main_thread.join();

    // timer.print("Kernel", p.n_reps);

    // Free memory
    
    // free(h_in_out);
    // free(h_flags);
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Kernel");
    timer.release("Deallocation");

    printf("Test Passed\n");
    // return 0;
}

// Main ------------------------------------------------------------------------------------------
int main(){

    float* A;
    float* B;
    float* C;

    float* A_device;
    float* B_device;
    float* C_device;

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

    CPU_GPU_Xor(A, B, C, 1, 1, 1,
                            A_Row, A_Column,
                            B_Row, B_Column,
                            C_Row, C_Column,
                            Bc_Host, NULL);

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
