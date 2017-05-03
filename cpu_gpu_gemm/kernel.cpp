#include "kernel.h"
#include "support/partitioner.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>


// CPU threads--------------------------------------------------------------------------------------
void CPU_GEMM(){

    int n_threads = 10;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
    
        cpu_threads.push_back(std::thread([=]() {}));
    }

    //Join the threads
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}


int main(){
    std::thread main_thread(CPU_GEMM);
    std::cout << "threadnum: " << main_thread.get_id() << std::endl;

    main_thread.join();
}


