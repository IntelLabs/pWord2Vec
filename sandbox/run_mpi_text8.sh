#!/bin/bash

# example run on 2 nodes, each node with 4 threads
# please specify the host file accordingly

nprocs=2

mpirun -np $nprocs ./mpi_job.sh

