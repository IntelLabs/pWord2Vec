#!/bin/bash

data=../data/text8
binary=../pWord2Vec_mpi

ncores=4
niters=2

#export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
numactl --interleave=all $binary -train $data -output vectors.txt -size 100 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab.txt -batch-size 17 -alpha 0.05

#note: batch-size is usually set to 2*window+1
