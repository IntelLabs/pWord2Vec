#!/bin/bash

printf "1. clean up workspace\n"
./clean.sh

printf "\n2. install hyperwords for evaluation\n"
hg clone -r 56 https://bitbucket.org/omerlevy/hyperwords

printf "\n3. build pWord2Vec\n"
make clean all

