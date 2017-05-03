import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

#np.set_printoptions(threshold=np.nan)
mod = SourceModule(open("binary_kernels.cu").read())
gemm = mod.get_function("gemm")
trans_kernel = mod.get_function("transpose")
prt_kernel = mod.get_function("prt")
#gemmr = mod.get_function("gemmr")

#A_SIZE = 10000 * 784
m = 10000
n = 784	
k = 4096
A_SIZE = m * n
B_SIZE = n * k
C_SIZE = m * k

A = np.ones(A_SIZE).astype(np.float32)
#for i in range(A_SIZE):
#    A[i] = i
T = np.empty(A_SIZE).astype(np.float32)
B = np.ones(B_SIZE).astype(np.float32)
#B = B * 1
#A = A * 2

C = np.empty(C_SIZE).astype(np.float32)
T2 = np.empty(A_SIZE).astype(np.float32)

A_gpu = cuda.mem_alloc(A.nbytes)
cuda.memcpy_htod(A_gpu, A)

T_gpu = cuda.mem_alloc(A.nbytes)
cuda.memcpy_htod(T_gpu, T)

T_gpu2 = cuda.mem_alloc(A.nbytes)

B_gpu = cuda.mem_alloc(B.nbytes)
cuda.memcpy_htod(B_gpu, B)

C_gpu = cuda.mem_alloc(C.nbytes)

"""
block_size = 16
block0 = (block_size, block_size, 1)
grid0 = (10000/block_size+1, 784/block_size+1,1)
#grid0 = (782/block_size+1, 10000/block_size+1,1)
trans_kernel(A_gpu,T_gpu,np.intc(10000), np.intc(784), block=block0, grid = grid0 );
"""

block_size = 16
block0 = (block_size, block_size, 1)
grid0 = (n/block_size+1, m/block_size+1,1)
trans_kernel(A_gpu,T_gpu,np.intc(m), np.intc(n),block=block0, grid = grid0 );


grid1 = (m/block_size+1, n/block_size+1,1)
trans_kernel(T_gpu,T_gpu2,np.intc(n), np.intc(m),block=block0, grid = grid1 );



cuda.memcpy_dtoh(T2, T_gpu2)

for i in range(A_SIZE):
	if T2[i] != A[i]:
		print i





#TODO: SET BLOCK SIZE
block_sizeM = 16
block_sizeN = 4  
block = (block_sizeM,1,1)
grid = (m / block_sizeM+1, k / block_sizeN+1,1)
gemm(T_gpu, B_gpu, C_gpu, np.intc(m), np.intc(n), np.intc(k), block= block, grid=grid)



A_gpu.free()
T_gpu.free()
B_gpu.free()


cuda.memcpy_dtoh(C, C_gpu)

print C


"""
f = open('inc.txt','w')
for i in range(0,10000):
    for j in range(0,4096):
        if C[i*4096+j] != 784:
            value = ('row',i,'col',j,'val',C[i*4096+j])
            s=str(value)
            f.write(s)
            f.write("\n")
f.close() """


C_gpu.free()





