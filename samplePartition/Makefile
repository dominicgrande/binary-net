OPT_GCC = -g 

OPT = -D_XOPEN_SOURCE=700
LIB = -lrt -pthread

all: cons prod

main: matrixMul serialMatrixMul
	nvcc $(OPT_GCC) serialMatrixMul.o matrixMul.o main.cu -o mainRun $(LIB)

matrixMul: matrixMul.cu
	nvcc $(OPT_GCC) $(OPT) -o matrixMul shm-posix-consumer.cu  $(LIB)

serialMatrixMul: serialMatrixMul.cu
	nvcc $(OPT_GCC) $(OPT) -o serialMatrixMul shm-posix-producer.cu  $(LIB)

run: main
	./mainRun

clean:
	rm -f cons prod trial
	rm -f *.o