#include <iostream>
#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast< TimeT> 
                            (std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 16 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int ldx, int ldy, int ldc, int M, int N, int K, float* A, float* B, float* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      float cij = C[i+j*ldy];
      for (int k = 0; k < K; ++k)
	    cij += A[i+k*ldc] * B[k+j*ldy];
      C[i+j*ldy] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int ldx, int ldy, int ldc, float* A, float* B, float* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < ldx; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < ldy; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < ldc; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, ldx-i);
	int N = min (BLOCK_SIZE, ldy-j);
	int K = min (BLOCK_SIZE, ldc-k);

	/* Perform individual block dgemm */
	do_block(ldx, ldy, ldc, M, N, K, A + k + i*ldc, B + k + j*ldy, C + i + j*ldy);
      }
}

int main(){

  float* A = new float[128 * 10000] ;
  float* B = new float[4096 * 128] ;
  float* C = new float[4096 * 10000] ;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  square_dgemm(10000, 4096, 128, A, B, C);
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}