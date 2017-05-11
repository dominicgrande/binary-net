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

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

// void CPU_GPU_Gemm(float * A, float* At, float * B, float * C, float alpha,
//                   int A_Row, int A_Column,
//                   int B_Row, int B_Column,
//                   int C_Row, int C_Column,
//                   int B_Host_ptr, int C_Host_ptr){

//     // Timer        timer;

//     // float * A = (float*)A_ptr;
//     // float * B = (float*)B_ptr;
//     // float * C = (float*)C_ptr;
//     // float * B_Host = (float*)B_Host_ptr;
//     // float * C_Host = (float*)C_Host_ptr;

//     // Allocate
//     // timer.start("Allocation");
//     cudaStream_t kernel_stream;
//     cudaStream_t data_stream;
//     cudaStreamCreate(&kernel_stream);
//     cudaStreamCreate(&data_stream);

//     const int A_GPU_Row     = (int)( A_Row * alpha);
//     const int A_CPU_Row     = A_Row - A_GPU_Row;

//     float* temp_A_Host;
//     int temp_A_Host_Size = sizeof(float)*A_CPU_Row*A_Column;
//     // temp_A_Host = (float *)malloc(temp_A_Host_Size);
//     cudaMallocHost(&temp_A_Host,temp_A_Host_Size);

//     // timer.stop("Allocation");
//     // timer.print("Allocation", 1);


//     // timer.start("Kernel Call");
//     //Changed the A_GPU_Row start with altered alpha value
//     cudaMemcpyAsync(temp_A_Host,&A[(A_GPU_Row)* A_Column], temp_A_Host_Size ,cudaMemcpyDeviceToHost, data_stream);
//     call_GPU_Kernel(A_Column, A_GPU_Row, B_Column, B_Row,
//                                  A_GPU_Row, C_Column, B, A, C,At, kernel_stream);

//     if (alpha<1.0){


//         // NONE STREAM
//         // cudaMemcpy(temp_A_Host, &A[(A_GPU_Row)* A_Column], sizeof(float)*(int) (A_CPU_Row*A_Column), cudaMemcpyDeviceToHost);

//         // STREAM
//         cudaStreamSynchronize (data_stream);
//         cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
//                 A_CPU_Row, B_Column, A_Column, 1, temp_A_Host, A_Column, B_Host, B_Column, 0.0, C_Host, B_Column);
//         cudaMemcpyAsync(&C[A_GPU_Row * C_Column], C_Host, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice, data_stream);
//     }

//     cudaDeviceSynchronize();
//     // timer.stop("Kernel Call");
//     // timer.print("Kernel Call", 1);

//     // timer.start("Deallocation");

//     // free(temp_A_Host);
//     cudaFreeHost(temp_A_Host);
//     cudaStreamDestroy(kernel_stream);
//     cudaStreamDestroy(data_stream);

//     // timer.stop("Deallocation");
//     // timer.print("Deallocation", 1);
//     // timer.stop("Allocation");
//     // timer.print("Allocation", 1);

//     // Release timers
//     // timer.release("Allocation");
//     // timer.release("Kernel");
//     // timer.release("Deallocation");

// }

// void serialMatrixMultiply(float *A, float *B, float *C,
//                                      int numARows, int numAColumns,
//                                      int numBRows, int numBColumns,
//                                      int numCRows, int numCColumns,
//                                      int numAStart, int numAStop){

//     printf("numAStart: %d - numAStop: %d\n", numAStart, numAStop);
    
//     for(int i=numAStart; i<numAStop; ++i){
//         for(int j=0; j<numBColumns; ++j){
//             C[i*numBColumns+j]=0;


//             for(int k=0; k<numAColumns; ++k)
//                 C[i*numBColumns+j]=C[i*numBColumns+j]+(A[(i-numAStart)*numAColumns+k]*B[k*numBColumns+j]);
//         }
//     }
// }

// // Main ------------------------------------------------------------------------------------------
// int main(){

//     float* A;
//     float* At;
//     float* B;
//     float* C;

//     float* A_device;
//     float* At_device;
//     float* B_device;
//     float* C_device;

// //     float* A_device;
// //     float* B_device;
// //     float* C_device;

//     int A_Row, A_Column, B_Row, B_Column, C_Row, C_Column;

//     C_Row = A_Row;
//     C_Column = B_Column;
// // float alpha = 1.0;
//     float alpha = 0.8;
//     // float alpha = .0000;
//     // float alpha = 0.0001;

//     A = (float *)malloc(A_Row*A_Column*sizeof(float));
//     At = (float *)malloc(A_Row*A_Column*sizeof(float));
//     B = (float *)malloc(B_Row*B_Column*sizeof(float));
//     C = (float *)malloc(C_Row*C_Column*sizeof(float));


//     cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
//     cudaMalloc(&At_device, A_Row*A_Column*sizeof(float));
//     cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
//     cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

// //     cudaMalloc(&A_device, A_Row*A_Column*sizeof(float));
// //     cudaMalloc(&B_device, B_Row*B_Column*sizeof(float));
// //     cudaMalloc(&C_device, C_Row*C_Column*sizeof(float));

//     for (int i=0; i<B_Row*B_Column; i++)
//         B[i] = 0.5;
// //     cudaMemcpy(A_device, A, sizeof(float)*A_Row*A_Column, cudaMemcpyHostToDevice);
// //     cudaMemcpy(B_device, B, sizeof(float)*B_Row*B_Column, cudaMemcpyHostToDevice);

// //     printf("After memcpy\n");
//     printf("After memcpy\n");
    
//     // std::cout << measure<>::execution(CPU_GPU_Gemm(A_device,At_device, B_device, C_device, alpha,A_Row, A_Column, B_Row, B_Column, C_Row, C_Column, B,C )) << std::endl;
//     int timex [21];
//     int iteration = 1;
//     for(int j = 0; j<=20;  j += 1){
//         alpha = j * .05;
//         int timetemp = 0;
//         for(int i = 0; i<iteration; i++){
//             std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//             CPU_GPU_Gemm(A_device,At_device, B_device, C_device, alpha,
//                         A_Row, A_Column,
//                         B_Row, B_Column,
//                         C_Row, C_Column,
//                         B,C );
//             std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

//             timetemp += (int)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

//         }
//         timex[j] = timetemp/iteration;
//         std::cout << " time: " << timetemp/iteration << std::endl;
        
//     }
//     // std::cout << " time: " << timetemp/iteration << std::endl;
//     for(auto const& value: timex)
//     {
//         std::cout << value << ",";
//     }


// //     cudaMemcpy(C, C_device, sizeof(float)*C_Column*C_Row, cudaMemcpyDeviceToHost);

//     // for (int i=0; i<C_Column*10000; i++){
//     //   if( C[i] != 784*.5){
// //        std::cout << "WRONG: "<< "x: " <<i%4096 << " y: " << i/4096<< "  " << C[i] << std::endl;
//     //    }
//     // }
//     std::cout << "ALL GOOD" << std::endl;
    
// //     free(A);
// //     free(B);
// //     free(C);
    
// }
static int test_global = 0;

void CPU_GPU_Gemm_test(unsigned long A, int length){
                //, float* At, float * B, float * C, float alpha,
                //   int A_Row, int A_Column,
                //   int B_Row, int B_Column,
                //   int C_Row, int C_Column,
                //   int B_Host_ptr, int C_Host_ptr){

        printf("test global %d\n", test_global++);
        float* A_gpu = (float *) A;

        call_gpu_function(A_gpu, length);


}

PYBIND11_PLUGIN(gemm)
{
  pybind11::module m("gemm", "GPU Library");
//   m.def("CPU_GPU_Gemm", CPU_GPU_Gemm);
  m.def("CPU_GPU_Gemm_test", CPU_GPU_Gemm_test);
  
}




