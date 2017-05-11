CXX=nvcc
CXX_FLAGS=-std=c++11 -O3 -use_fast_math -Xptxas --verbose

ifndef CHAI_CUDA_LIB
    $(error CHAI_CUDA_LIB not defined. This environment variable must be defined to point to the location of the CUDA library)
endif
LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm

ifndef CHAI_CUDA_INC
    $(error CHAI_CUDA_INC not defined. This environment variable must be defined to point to the location of the CUDA header files)
endif
INC=-I$(CHAI_CUDA_INC)

DEP=kernel.cu xor.cpp kernel.h main.cpp kernel.cu support/common.h support/cuda-setup.h support/partitioner.h support/timer.h support/verify.h
SRC=main.cpp kernel.cu xor.cpp
EXE=pad

all:
	$(CXX) $(CXX_FLAGS) $(SRC) $(LIB) $(INC) -o $(EXE) -lblas 

clean:
	rm -f $(EXE)

