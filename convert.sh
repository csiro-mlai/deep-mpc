#!/bin/bash

out_dir=MP-SPDZ/Player-Data
test -e $out_dir || mkdir $out_dir

./full.py > $out_dir/Input-P0-0
