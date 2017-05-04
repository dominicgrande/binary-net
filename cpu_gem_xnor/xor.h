#ifndef XOR_H
#define XOR_H

unsigned int multiply_and_pop(unsigned int A, unsigned int B);

void multiplyMatrices(unsigned int* A, unsigned int* B, float* C, 
int rowFirst, int columnFirst, int rowSecond, int columnSecond);

unsigned int concatenate(float* array);

void concatenate_rows_serial(float* input, unsigned int* output, 
                                int row, int column);

void concatenate_cols_serial(float* input, unsigned int* output, 
                                int m, int n);

#endif