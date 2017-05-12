#include <iostream>
#include <cuda>

int main(){
    int* atomicVal;
    cudaMalloc(&atomicVal, sizeof(int));
    int startVal = 0;
    cudaMemcpy(atomicVal, startVal, sizeof(int), memcpyHostToDevice);

    int printOut = atomicAdd(atomicVal, 1);
    std::cout << "The atomic add value is: " << printOut;
}