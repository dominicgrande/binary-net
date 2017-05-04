#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <iostream>
#include <string.h>
#include <cblas.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

void CPU_GPU_Xnor(float * A, float * B, float * C, float alpha_1, float alpha_2, float alpha_3,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column,
                  float* B_Host, float* C_Host){

    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    float * Ac, Bc;

    // const int A_GPU_Row     = (int) A_Row * alpha;
    // const int A_CPU_Row     = A_Row - A_GPU_Row;

    cudaMalloc(&Ac, n*m*sizeof(float)/32);
    cudaMalloc(&Bc, n*k*sizeof(float)/32);
    timer.stop("Allocation");
    timer.print("Allocation", 1);


    timer.start("Kernel Call");

    // void call_GPU_concatenate_rows(int n, int m, float* A, float* Ac);
    // void call_GPU_concatenate_cols(int n, int m, int k, float* B, float* Bc);
    // void call_GPU_xnor(int n, int m, int k, float* Ac, float* Bc, float* C);
    //Changed the A_GPU_Row start with altered alpha value

    void call_GPU_concatenate_rows(A_Column, A_Row, A, Ac);
    printf("Made it after GPU kernel. Need sync\n");
    float* temp_A_Host;
    if (alpha<1){
        temp_A_Host = (float *)malloc(sizeof(float)*A_CPU_Row*A_Column);

        cudaMemcpy(temp_A_Host, &A[(A_GPU_Row)* A_Column], sizeof(float)*(int) (A_CPU_Row*A_Column), cudaMemcpyDeviceToHost);

        printf("Memcpy is no good.\n");

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A_CPU_Row, B_Column, A_Column, 1, temp_A_Host, A_Column, B_Host, B_Column, 0.0, C_Host, B_Column);

        // serialMatrixMultiply(temp_A_Host, B_Host, C_Host, 
                            // A_Row, A_Column,
                            // B_Row, B_Column,
                            // C_Row, C_Column,
                            // A_GPU_Row, A_Row);
        free(temp_A_Host);

    }
    

    // Launch CPU threads
    // std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.n, p.m, p.pad, p.n_threads, p.n_gpu_threads, n_tasks, p.alpha);

    cudaDeviceSynchronize();
    cudaMemcpy(&C[A_GPU_Row * C_Column], C_Host, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice);
    timer.stop("Kernel Call");
    timer.print("Kernel Call", 1);
    // main_thread.join();

    // timer.print("Kernel", p.n_reps);

    // Free memory
    timer.start("Deallocation");
    // free(h_in_out);
    // free(h_flags);
    cudafree(Ac); 
    cudafree(Bc); 
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
    A_Column = 784;
    B_Column = 4096;
    B_Row = 784;

    C_Row = A_Row;
    C_Column = B_Column;
    float alpha = .0101;
    // float alpha = .0000;
    // float alpha = 0.0001;

    A = (float *)malloc(A_Row*A_Column*sizeof(float));
    B = (float *)malloc(B_Row*B_Column*sizeof(float));
    C = (float *)malloc(C_Row*C_Column*sizeof(float));

    cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
    cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
    cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

    for (int i=0; i<A_Row*A_Column; i++)
        A[i] = 1.0;

    for (int i=0; i<B_Row*B_Column; i++)
        B[i] = 1.0;

    cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float)*B_Row*B_Column, cudaMemcpyHostToDevice);

    printf("After memcpy\n");

    CPU_GPU_Gemm(A_device, B_device, C_device, alpha,
                  A_Row, A_Column,
                  B_Row, B_Column,
                  C_Row, C_Column,
                  B, C);

    cudaMemcpy(C, C_device, sizeof(float)*C_Column*C_Row, cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

   for (int i=0; i<C_Column*10000; i++){
       if( C[i] != 784){

        std::cout << "WRONG: "<< "x: " <<i%4096 << " y: " << i/4096<< "  " << C[i] << std::endl;

       }
   }
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
