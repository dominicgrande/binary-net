main: matrixMul serialMatrixMul
	nvcc serialMatrixMul.o matrixMul.o main.cu -o mainRun 

matrixMul: matrixMul.cu
	nvcc -o matrixMul matrixMul.cu  

serialMatrixMul: serialMatrixMul.cu
	nvcc -o serialMatrixMul serialMatrixMul.cu  

run: main
	./mainRun

clean:
	rm -f cons prod trial
	rm -f *.o
