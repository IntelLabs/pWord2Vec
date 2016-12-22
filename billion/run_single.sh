#!/bin/bash

data=../data/1b
binary=../pWord2Vec

# set this number to #logical cores of your machine (with hyper-threading if available)
ncores=72 # 72 is for Intel Xeon E5-2697 v4 (BDW)

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
numactl --interleave=all $binary -train $data -save-vocab vocab.txt -output vectors.txt -size 300 -window 5 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter 5 -min-count 2 -batch-size 11

#tips: run the above line once with "-save-vocab" to generate vocabulary, then use "-read-vocab" to load vocabulary directly for later runs.
