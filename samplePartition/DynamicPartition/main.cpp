#include <iostream>
#include <cuda>

int main(){
    int* val;
    cudaMalloc(&val, sizeof(int));
    
    cudaMemset(val, 1, 1);

    int *output;
    cudaMemcpy(output, val, sizeof(int), cudamemcpyDeviceToHost);
}