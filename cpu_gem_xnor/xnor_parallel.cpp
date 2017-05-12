#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <vector>
#include <algorithm>

    

inline unsigned int multiply_and_pop(unsigned int A, unsigned int B){
    // unsigned int x_or_value = ;
    // unsigned int x =     // std::cout << "Bit count value: " << x << std::endl;
    return __builtin_popcount(A ^ B);
}

void multiplyMatrices(unsigned int* A, unsigned int* B, float* C,int thread, 
int rowFirst, int columnFirst, int rowSecond, int columnSecond)
{
	int i, j, k, temp, q,p, m, n;
    int value = columnFirst * 32;
	// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
    // int end = (thread+1)*rowFirst;
	for(i = 0; i < rowFirst; i= i + 16)
	{
        q = i * columnFirst;
		for(j = 0; j < columnSecond; j = j + 16)
		{
            for(m = 0; m < 16; m++){
                for(n = 0; n < 16; n++){


                p = j * columnFirst;
                    for(k=0; k<columnFirst; ++k)
                    {
                        // temp += multiply_and_pop(A[i*columnFirst+k], B[k*columnSecond+j]);
                        temp += __builtin_popcount(A[q+k]^ B[p+k]);
                    }
                    C[i*columnSecond+p] = -(2*(float)temp-value);
                }
            }
		}
	}
}

unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

void concatenate_rows_serial(float* input, unsigned int* output, 
                                int row, int column){
    for (int i=0; i<(row*column)/32; i++){
        output[i] = concatenate(&input[i*32]);
    }   

    // std::cout <<  "The value of the output array for rows is: " << std::endl;
    // for (int i=0; i<32; i++)
    //     std::cout <<std::setbase(16) << std::showbase <<(int)output[i] << " " ;

    // std::cout << std::endl;
}


void concatenate_cols_serial(float* input, unsigned int* output, 
                                int m, int n){
    
    float * array = new float[32];
    for (int j=0; j<n; j++){
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) 
                array[k] = input[j + n*(i+k)];
            output[j+n*i/32]=concatenate(array); 
        } 
    }
    //  std::cout <<  "The value of the output array for cols is: " << std::endl;
    // for (int i=0; i<32; i++)
    //     std::cout <<std::setbase(16) << std::showbase <<(int)output[i] << " ";
    
    // std::cout << std::endl;
    delete[] array;
}

int main(){
    float* A;
    A = new float[10000*4096];
    float* B;
    B = new float[4096*4096];

    for (int i=0; i<10000*4096; i++)
        A[i] = 1.0;

    for (int j=0; j<4096*4096; j++)
        B[j] = -1.0;



    unsigned int* A_Shrunk = new unsigned int[128 * 10000];
    unsigned int* B_Shrunk = new unsigned int[128 * 4096];

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    concatenate_rows_serial(A, A_Shrunk, 10000, 4096);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();
    concatenate_cols_serial(B, B_Shrunk, 4096, 4096);
    std::chrono::steady_clock::time_point end3= std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count() << std::endl;

    float* C_Shrunk = new float[10000*4096];

    // for (int i=0; i<32*32; i++)
    // {
    //     C_Shrunk[i] = 0;
    // }

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();

    int n_threads = 1;
    std::vector<std::thread> cpu_threads;

    for(int i = 0; i < n_threads; i++) {
        cpu_threads.push_back(std::thread(multiplyMatrices,B_Shrunk, A_Shrunk, C_Shrunk,i, 1000, 128, 128, 4096));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
    std::chrono::steady_clock::time_point end2= std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << std::endl;



    // std::cout << "C values" << std::endl;
    // for (int x=0; x<32; x++)
    // {
    //     for (int y=0; y<32; y++)
    //         std::cout << std::setbase(16) << std::showbase <<(int)(C_Shrunk[x*32+y]) << " " ;

    //     std::cout << std::endl;
    // }
}

