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




