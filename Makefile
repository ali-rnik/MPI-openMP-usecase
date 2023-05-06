all: mpi openmp
	
mpi:
	mpic++ -g mpi.cpp -fopenmp -o mpi
openmp:
	g++ -g openmp.cpp -fopenmp -o openmp

clean:
	rm mpi openmp