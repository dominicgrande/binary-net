CXX=nvcc -ccbin g++-4.9 -use_fast_math -O3 -Xptxas --verbose
CXX_FLAGS=-std=c++11 -shared --compiler-options -fPIC 


ifndef CHAI_CUDA_LIB
    $(error CHAI_CUDA_LIB not defined. This environment variable must be defined to point to the location of the CUDA library)
endif
LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm -L/usr/lib/python2.7/config-x86_64-linux-gnu -L/usr/lib

ifndef CHAI_CUDA_INC
    $(error CHAI_CUDA_INC not defined. This environment variable must be defined to point to the location of the CUDA header files)
endif
INC=-I$(CHAI_CUDA_INC) -I./include -I/usr/include/python2.7 -I/usr/include/python2.7 -DNDEBUG -g 
DEP=kernel.cpp kernel.h main.cpp kernel.cu support/common.h support/cuda-setup.h support/partitioner.h support/timer.h support/verify.h
SRC=gemm.cpp kernel.cu
EXE=../BinaryNet/Run-time/gemm.so

all:
	$(CXX) $(CXX_FLAGS) $(SRC) $(LIB) $(INC) -Xlinker -export-dynamic -lpython2.7 -lpthread -ldl  -lutil -lrt -lblas -o $(EXE) 
clean:
	rm -f $(EXE)

