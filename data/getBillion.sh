#!/bin/bash

echo "Fetch the 1 billion word language modeling benchmark"

if [ ! -e 1-billion-word-language-modeling-benchmark-r13output.tar.gz ]; then
  echo "Downloading ..."
  wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

if [ ! -d 1-billion-word-language-modeling-benchmark-r13output ]; then
  echo "Unzipping ..."
  tar xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

echo "Aggregating training text ..."
cat 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 > 1b
cat 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/* >> 1b

echo "clean up ..."
rm -rf 1-billion-word-language-modeling-benchmark-r13output
