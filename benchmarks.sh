#!/bin/bash

cd MP-SPDZ

./compile.py -CDR 64 -K '' ../$1.mpc 20 128 8 trunc_pr print_losses
mkdir logs
Scripts/emulate.sh $1-20-128-8-trunc_pr-print_losses -IF /tmp/cifar10-Input
