#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

void CPU_GPU_Gemm(float * A, float * B, float * C,float alpha) 
                  int A_Row, int A_Column,
                  int B_Row, int B_Column,
                  int C_Row, int C_Column);{

    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    const int alpha     = 0;

    float*    h_in_out = (float*)malloc(in_size * sizeof(float));
    float*    d_in_out;

    cudaStatus = cudaMalloc((void**)&d_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T));
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));

    T *h_in_backup = (T *)malloc(in_size * sizeof(T));

    cudaDeviceSynchronize();

    timer.stop("Allocation");
    timer.print("Allocation", 1);


    // Initialize
    timer.start("Initialization");
    memset(h_flags, 0, n_flags * sizeof(atomic_int));

    timer.stop("Initialization");
    timer.print("Initialization", 1);

    // timer.start("Copy To Device");
    // cudaStatus = cudaMemcpy(d_flags, h_flags, n_flags * sizeof(int), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();
    // CUDA_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);

    cudaStatus = call_Padding_kernel(p.n_gpu_blocks, p.n_gpu_threads, p.n, p.m, p.pad, n_tasks, p.alpha);

    // Launch CPU threads
    std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.n, p.m, p.pad, p.n_threads, p.n_gpu_threads, n_tasks, p.alpha);

    cudaDeviceSynchronize();
    main_thread.join();

    timer.print("Kernel", p.n_reps);

    // Free memory
    timer.start("Deallocation");
    // free(h_in_out);
    // free(h_flags);
    // cudaStatus = cudaFree(d_in_out);
    // cudaStatus = cudaFree(d_flags);
    cudaDeviceSynchronize();
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
    return 0;
}

// Main ------------------------------------------------------------------------------------------
int main(){

    CPU_GPU_Gemm(float * A, float * B, float * C);
}