#!/bin/sh

code=../hyperwords
src_model=vectors.txt
model=pWord2Vec

cp $src_model $model.words

python $code/hyperwords/text2numpy.py $model.words

echo "WS353 Results"
echo "-------------"
python $code/hyperwords/ws_eval.py embedding $model $code/testsets/ws/ws353.txt
echo

echo "Google Analogy Results"
echo "----------------------"
python $code/hyperwords/analogy_eval.py embedding $model $code/testsets/analogy/google.txt
echo

