#!/bin/bash

wget -nc http://yann.lecun.com/exdb/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz || exit 1

for i in *.gz; do
    gunzip $i
done
