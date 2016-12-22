CC = icpc
MPICC = mpiicpc
CFLAGS = -std=c++11 -qopenmp -O3 -D USE_MKL -mkl=sequential -Wall -xhost

all: pWord2Vec pWord2Vec_mpi 

pWord2Vec: pWord2Vec.cpp
	$(CC) pWord2Vec.cpp -o pWord2Vec $(CFLAGS)
pWord2Vec_mpi: pWord2Vec_mpi.cpp
	$(MPICC) pWord2Vec_mpi.cpp -o pWord2Vec_mpi $(CFLAGS)
clean:
	rm -rf pWord2Vec pWord2Vec_mpi 
