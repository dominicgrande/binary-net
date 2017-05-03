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

void CPU_GPU_Gemm(int  A_ptr, int  B_ptr, int  C_ptr, int alpha,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column,
                  int B_Host_ptr, int C_Host_ptr){

    Timer        timer;
    cudaError_t  cudaStatus;

    float * A = (float*)A_ptr;
    float * B = (float*)B_ptr;
    float * C = (float*)C_ptr;
    float * B_Host = (float*)B_Host_ptr;
    float * C_Host = (float*)C_Host_ptr;

    // Allocate
    timer.start("Allocation");

    const int A_GPU_Row     = (int) A_Row * alpha;
    const int A_CPU_Row     = A_Row - A_GPU_Row;

    timer.stop("Allocation");
    timer.print("Allocation", 1);


    timer.start("Initialization");

    timer.stop("Initialization");
    timer.print("Initialization", 1);

    timer.start("Kernel Call");
    //Changed the A_GPU_Row start with altered alpha value
    call_GPU_Kernel(A_Column, A_GPU_Row, B_Column, B_Row,
                                 A_GPU_Row, C_Column, B, A, C);
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
    // cudaStatus = cudaFree(d_in_out);
    // cudaStatus = cudaFree(d_flags);
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Initialization");
    timer.release("Copy To Device");
    timer.release("Kernel");
    timer.release("Copy Back and Merge");
    timer.release("Deallocation");

    printf("Test Passed\n");
    // return 0;
}

void serialMatrixMultiply(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int numAStart, int numAStop){

    printf("numAStart: %d - numAStop: %d\n", numAStart, numAStop);
    
    for(int i=numAStart; i<numAStop; ++i){
        for(int j=0; j<numBColumns; ++j){
            C[i*numBColumns+j]=0;


            for(int k=0; k<numAColumns; ++k)
                C[i*numBColumns+j]=C[i*numBColumns+j]+(A[(i-numAStart)*numAColumns+k]*B[k*numBColumns+j]);
        }
    }
}

// Main ------------------------------------------------------------------------------------------
// int main(){

//     float* A;
//     float* B;
//     float* C;

//     float* A_device;
//     float* B_device;
//     float* C_device;

//     int A_Row, A_Column, B_Row, B_Column, C_Row, C_Column;

//     A_Row = 10000;
//     A_Column = 784;
//     B_Column = 4096;
//     B_Row = 784;

//     C_Row = A_Row;
//     C_Column = B_Column;
//     float alpha = .001;
//     // float alpha = .0000;
//     // float alpha = 0.0001;

//     A = (float *)malloc(A_Row*A_Column*sizeof(float));
//     B = (float *)malloc(B_Row*B_Column*sizeof(float));
//     C = (float *)malloc(C_Row*C_Column*sizeof(float));

//     cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
//     cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
//     cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

//     for (int i=0; i<A_Row*A_Column; i++)
//         A[i] = 1.0;

//     for (int i=0; i<B_Row*B_Column; i++)
//         B[i] = 2.0;

//     cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
//     cudaMemcpy(B_device, B, sizeof(float)*B_Row*B_Column, cudaMemcpyHostToDevice);

//     printf("After memcpy\n");

//     CPU_GPU_Gemm(A_device, B_device, C_device, alpha,
//                   A_Row, A_Column,
//                   B_Row, B_Column,
//                   C_Row, C_Column,
//                   B, C);

//     cudaMemcpy(C, C_device, sizeof(float)*C_Column*C_Row, cudaMemcpyDeviceToHost);

//     cudaFree(A_device);
//     cudaFree(B_device);
//     cudaFree(C_device);

//    for (int i=0; i<C_Column*10000; i++){
//        if( C[i] != 784*2){
//         if( i % C_Column == 0){
//             std::cout << "WRONG: "<< "x: " <<i%4096 << " y: " << i/4096<< "  " << C[i] << std::endl;
//         }

//        }
//    }
//     std::cout << "ALL GOOD" << std::endl;
    
//     free(A);
//     free(B);
//     free(C);
    
	
// // for (int i=0; i<C_Row; i++){
//     //     for (int j=0; j<C_Column; j++){
//     //         printf("%f ", C[i*C_Column+j]);
//     //     }
//     //     printf("\n");
//     // }
// }

// namespace py = pybind11;

PYBIND11_PLUGIN(gemm)
{
  pybind11::module m("gemm", "GPU Library");
  m.def("CPU_GPU_Gemm", CPU_GPU_Gemm);
  
}



