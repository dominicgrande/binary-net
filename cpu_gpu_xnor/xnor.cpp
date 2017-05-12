#include "kernel.h"
#include "support/timer.h"
#include "support/verify.h"

#include <iostream>
#include <string.h>
#include <cblas.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

//Libaries neccesary for creating shared library for python
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Cuda Libaries
#include <cuda_runtime.h>
#include <cuda.h>

// Libaries for timing the code
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

//host memory for data trasfer for collaboration
static float* A_Host;
static float* C_Host;

// Weight array
static float* B_Host;

//array for packed data
static unsigned int* Ac;
static unsigned int* Bc;

//For transformed A array
static float* At;

//Libaries variabes
static int pinned_memory;
static float alpha;

void CPU_GPU_Xnor(unsigned long A_ptr, unsigned long B_ptr, unsigned long C_ptr,
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column){

    //fetch the pointer values from the interface
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

        memset(C_Host,1,sizeof(float)* A_CPU_Row * C_Column);
        cudaStreamSynchronize (data_stream);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                A_CPU_Row, B_Column, A_Column, -2, A_Host, A_Column, B_Host, B_Column, -4096, C_Host, B_Column);

        cudaMemcpyAsync(&C[A_GPU_Row * C_Column], C_Host, sizeof(float)*A_CPU_Row*C_Column, cudaMemcpyHostToDevice, data_stream);
    }

    cudaDeviceSynchronize();
    
    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(data_stream);




}

static int test_global = 0;


void Load_Weights(pybind11::array_t<float> vec)
{

    auto info_i = vec.request();
    B_Host = (float*)info_i.ptr;

}
void initLibrary(float alpha_in, int pinned_memory_input,
                int A_Row, int A_Column,
                int B_Row, int B_Column,
                int C_Row, int C_Column)
{
    //Set library variables
    pinned_memory = pinned_memory_input;
    alpha = alpha_in;


    // Allocate memory on the CPU & GPU
    int height = A_Row -  (int)( A_Row * alpha);

    if(pinned_memory == 1){
        cudaMallocHost(&A_Host,height * A_Column * sizeof(float));
        cudaMallocHost(&C_Host,height * C_Column * sizeof(float));
    }
    else{
        A_Host = new float [height*A_Column];
        C_Host = new float [height*C_Column];
    }
    cudaMalloc(&At, A_Row*A_Column*sizeof(float));
    cudaMalloc(&Ac,A_Row * A_Column/32 * sizeof(unsigned int));
    cudaMalloc(&Bc,B_Row * B_Column/32 * sizeof(unsigned int));


}

void deallocateMemory(int alpha,
                int A_Row, int A_Column,
                int B_Row, int B_Column,
                int C_Row, int C_Column)
{
    // A_host = (float *)malloc(A_Row*A_Column*sizeof(float));
    // C_host = (float *)malloc(C_Row*C_Column*sizeof(float));

    if(pinned_memory == 1){
        cudaFreeHost(A_Host);
        cudaFreeHost(C_Host);
    }
    else{
        delete [] A_Host;
        delete [] C_Host;
    }
    cudaFree(At);
    cudaFree(Ac);
    cudaFree(Bc);
  
}
void CPU_GPU_Gemm_test(unsigned long A, int length){

        // printf("test global %d\n", test_global++);
        // printf("w value: %f", W_host[1]);
        float* A_gpu = (float *) A;

        // call_gpu_function(A_gpu, length);


}

PYBIND11_PLUGIN(xnor)
{
  pybind11::module m("xnor", "GPU Library");
  m.def("CPU_GPU_Xnor", CPU_GPU_Xnor);
  m.def("Load_Weights", Load_Weights);
  m.def("initLibrary", initLibrary);
  m.def("deallocateMemory", deallocateMemory);
  
}

