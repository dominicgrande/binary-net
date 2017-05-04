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

void call_GPU_Kernel(int numAColumns, int numARows, int numBColumns, int numBRows,
                        int numCRows, int numCColumns, float *weights, float *x, float* output);

void serialMatrixMultiply(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int numAStart, int numAStop);

#endif 
