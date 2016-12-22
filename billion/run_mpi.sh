#!/bin/bash

# example run on 4 Intel BDW nodes, and each node with 72 threads
# please specify the host file accordingly

nprocs=4

mpirun -np $nprocs ./mpi_job.sh
