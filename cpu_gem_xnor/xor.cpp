#include "xor.h"
#include <iostream>
#include <iomanip>

unsigned int multiply_and_pop(unsigned int A, unsigned int B){
    unsigned int x_or_value = A ^ B;
    unsigned int x = __builtin_popcount(x_or_value);
    std::cout << "Bit count value: " << x << std::endl;
    return x;
}

void multiplyMatrices(unsigned int* A, unsigned int* B, float* C, 
int rowFirst, int columnFirst, int rowSecond, int columnSecond)
{
	int i, j, k;
	// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
	for(i = 0; i < rowFirst; ++i)
	{
		for(j = 0; j < columnSecond; ++j)
		{
			for(k=0; k<columnFirst; ++k)
			{
				C[i*columnSecond+j] += multiply_and_pop(A[i*columnFirst+k], B[k*columnSecond+j]);
			}
            C[i*columnSecond+j] = -(2*(float)C[i*columnSecond+j]-32*columnFirst);
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

    std::cout <<  "The value of the output array for rows is: " << std::endl;
    for (int i=0; i<32; i++)
        std::cout <<std::setbase(16) << std::showbase <<(int)output[i] << " " ;

    std::cout << std::endl;
}


void concatenate_cols_serial(float* input, unsigned int* output, 
                                int m, int n){
    
    for (int j=0; j<n; j++){
        float * array = new float[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) 
                array[k] = input[j + n*(i+k)];
            output[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
     std::cout <<  "The value of the output array for cols is: " << std::endl;
    for (int i=0; i<32; i++)
        std::cout <<std::setbase(16) << std::showbase <<(int)output[i] << " ";
    
    std::cout << std::endl;
}

int main(){
    float* A;
    A = new float[32*64];
    float* B;
    B = new float[32*64];

    for (int i=0; i<32*64; i++)
        A[i] = 1.0;

    for (int j=0; j<32*64; j++)
        B[j] = -1.0;

    unsigned int* A_Shrunk = new unsigned int[64];
    unsigned int* B_Shrunk = new unsigned int[64];

    concatenate_rows_serial(A, A_Shrunk, 32, 64);
    concatenate_cols_serial(B, B_Shrunk, 64, 32);

    float* C_Shrunk = new float[32*32];

    for (int i=0; i<32*32; i++)
    {
        C_Shrunk[i] = 0;
    }

    multiplyMatrices(B_Shrunk, A_Shrunk, C_Shrunk, 32, 2, 2, 32);
    std::cout << "C values" << std::endl;
    for (int x=0; x<32; x++)
    {
        for (int y=0; y<32; y++)
            std::cout << C_Shrunk[x*32+y]) << " " ;
            // std::cout << std::setbase(16) << std::showbase <<(int)(C_Shrunk[x*32+y]) << " " ;

        std::cout << std::endl;
    }
}

