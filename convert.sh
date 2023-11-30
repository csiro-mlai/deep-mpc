#!/bin/bash

out_dir=MP-SPDZ/Player-Data
test -e $out_dir || mkdir $out_dir

./full.py $1 > $out_dir/Input-P0-0
