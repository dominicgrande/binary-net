/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    const int alpha     = 0;

    T *    h_in_out = (T *)malloc(in_size * sizeof(T));
    T *    d_in_out;

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
