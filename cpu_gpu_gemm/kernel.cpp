#include "kernel.h"
#include "support/partitioner.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>


// CPU threads--------------------------------------------------------------------------------------
void CPU_GEMM(){

    int n_threads = 4;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
    
        cpu_threads.push_back(std::thread([=]() {

        }));
    }

    //Join the threads
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}


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


