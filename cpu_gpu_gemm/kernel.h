#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include <atomic>
#include "support/common.h"

void run_cpu_threads(T *matrix_out, T *matrix, std::atomic_int *flags, int n, int m, int pad, int num_threads, int ldim, int n_tasks, float alpha
#ifdef CUDA_8_0
    , std::atomic_int *worklist
#endif
    );

cudaError_t call_Padding_kernel(int blocks, int threads, int n, int m, int pad, int n_tasks, float alpha, 
    T *matrix_out, T *matrix, int *flags
#ifdef CUDA_8_0
    , int l_mem_size, int *worklist
#endif
		);

cudaError_t call_GPU_Kernel(int blocks, int threads, int numAColumns, int numARows, int numBColumns, int numBRows
     float alpha, int numCRows, int numCColumns, float *weights, float *x, float* output)

#endif KERNEL_H
