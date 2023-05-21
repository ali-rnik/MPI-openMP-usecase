all: mpi openmp openmpC
	
mpi:
	mpic++ -g mpi.cpp -fopenmp -o mpi
openmp:
	g++ -g openmp.cpp -fopenmp -o openmp
openmpC:
	gcc -g openmp.c -fopenmp -o openmpC

clean:
	rm mpi openmp