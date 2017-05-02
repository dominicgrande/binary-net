main: matrixMul serialMatrixMul
	nvcc serialMatrixMul.o matrixMul.o main.cu -o mainRun 

matrixMul: matrixMul.cu
	nvcc -o matrixMul.o matrixMul.cu  

serialMatrixMul: serialMatrixMul.cu
	nvcc -o serialMatrixMul.o serialMatrixMul.cu  

run: main
	./mainRun

clean:
	rm -f cons prod trial
	rm -f *.o
