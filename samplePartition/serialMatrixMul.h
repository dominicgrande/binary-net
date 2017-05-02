#ifndef SERIAL_MATRIX_MUL_H
#define SERIAL_MATRIX_MUL_H
void serialMatrixMultiply(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int numAStart);

#endif