#!/bin/bash

wget -nc -q http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz

for i in *.gz; do
    gunzip $i
done
