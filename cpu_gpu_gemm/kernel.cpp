#include "kernel.h"
#include "support/partitioner.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>


// CPU threads--------------------------------------------------------------------------------------
// void CPU_GEMM(XYZ *in, XYZ *outp, int n_tasks, float alpha, int n_threads, int n_gpu_threads, int in_size_i, int in_size_j,
//     int out_size_i, int out_size_j
// #ifdef CUDA_8_0
//     , std::atomic_int *worklist
// #endif
//     ){


//     std::vector<std::thread> cpu_threads;
//     for(int k = 0; k < n_threads; k++) {
//         cpu_threads.push_back(std::thread([=]() {

// #ifdef CUDA_8_0
//             Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads, worklist);
// #else
//             Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads);
// #endif

//             const int wg_in_J = divceil(out_size_j, n_gpu_threads);
//             const int wg_in_I = divceil(out_size_i, n_gpu_threads);

//             for(int t = cpu_first(&p); cpu_more(&p); t = cpu_next(&p)) {
//                 const int my_s1 = t / wg_in_J;
//                 const int my_s0 = t % wg_in_J;

//                 int Row = my_s1 * n_gpu_threads;
//                 int Col = my_s0 * n_gpu_threads;
//                 T   bi;
//                 T   bj;
//                 T   mui, muj;

//                 for(int i = Row; i < Row + n_gpu_threads; i++) {
//                     mui = i / (T)(out_size_i - 1);
//                     for(int j = Col; j < Col + n_gpu_threads; j++) {
//                         muj = j / (T)(out_size_j - 1);
//                         if(i < out_size_i && j < out_size_j) {
//                             XYZ out = {0, 0, 0};
// #pragma unroll
//                             for(int ki = 0; ki <= in_size_i; ki++) {
//                                 bi = BezierBlend(ki, mui, in_size_i);
// #pragma unroll
//                                 for(int kj = 0; kj <= in_size_j; kj++) {
//                                     bj = BezierBlend(kj, muj, in_size_j);
//                                     out.x += (in[ki * (in_size_j + 1) + kj].x * bi * bj);
//                                     out.y += (in[ki * (in_size_j + 1) + kj].y * bi * bj);
//                                     out.z += (in[ki * (in_size_j + 1) + kj].z * bi * bj);
//                                 }
//                             }
//                             outp[i * out_size_j + j] = out;
//                         }
//                     }
//                 }
//             }

//         }));
//     }
//     std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
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
// int main(){
//     std::thread main_thread(CPU_GEMM);
//     std::cout << "threadnum: " << main_thread.get_id() << std::endl;

//     main_thread.join();
// }


