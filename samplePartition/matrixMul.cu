#define TILE_WIDTH 64

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  
  __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];


  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int row = by*blockDim.y+ty;
  int col = bx*blockDim.x+tx;
    
  float cValue = 0;
 
  for (int ph=0; ph<ceil((numAColumns)/float(TILE_WIDTH)); ++ph)
  {
    
    if ((row<numARows-1) && (ph*TILE_WIDTH+tx)<numAColumns) 
      Ads[ty][tx] = A[row*numAColumns + ph*TILE_WIDTH+tx];    
   else
      Ads[ty][tx] = 0;
    
    if ((ph*TILE_WIDTH+ty)<numBRows && (col<numBColumns))
      Bds[ty][tx] = B[(ph*TILE_WIDTH+ty)*numBColumns+col];
    else
      Bds[ty][tx] = 0;
    
    __syncthreads();
    
    for (int k=0; k<TILE_WIDTH; ++k)
    {
        cValue+=Ads[ty][k]*Bds[k][tx];
    }
    __syncthreads();
  }
  
  
  if ((row<numCRows) && (col<numCColumns))
  {
    C[row*numCColumns+col] = cValue;
  }
}
