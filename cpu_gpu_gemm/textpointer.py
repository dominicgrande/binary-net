import gemm
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

import numpy as np
import theano
import theano.tensor as T

import theano.misc.pycuda_init

import theano.sandbox.cuda as cuda
from theano.sandbox.cuda.basic_ops import host_from_gpu


A_Host = np.ones(200).astype(np.float32)

# A_gpu = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(A_Host))
A_gpu = cuda.CudaNdarray.zeros((200,))
# A_gpu = cuda.mem_alloc(A_Host.nbytes)
# cuda.memcpy_htod(A_gpu, A_Host)

A_gpu_pointer = A_gpu.gpudata

gemm.CPU_GPU_Gemm_test(A_gpu_pointer, 200)
gemm.CPU_GPU_Gemm_test(A_gpu_pointer, 200)
gemm.CPU_GPU_Gemm_test(A_gpu_pointer, 200)
gemm.CPU_GPU_Gemm_test(A_gpu_pointer, 200)

# cuda.memcpy_dtoh(A_Host, A_gpu)
B_host = np.array(A_gpu)  
print B_host