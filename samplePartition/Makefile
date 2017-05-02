main: matrixMul serialMatrixMul
	nvcc serialMatrixMul.o matrixMul.o main.cu -o mainRun 

matrixMul: matrixMul.cu
	nvcc -c matrixMul.cu  

serialMatrixMul: serialMatrixMul.cu
	nvcc -c serialMatrixMul.cu  

run: main
	./mainRun

clean:
	rm -f mainRun
	rm -f *.o
