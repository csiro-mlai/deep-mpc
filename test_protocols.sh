#!/bin/bash

./full.py 10 > MP-SPDZ/Player-Data/d-P0-0

opts="16 1dense rate.1 mini"

run_opt="-IF Player-Data/d -M" ./run-local.sh dm10 A prob 1 3 $opts

for i in sh2 dm3; do
    run_opt="-IF Player-Data/d -M" ./run-local.sh $i A prob 1 10 $opts
done

for i in emul sh3 mal4 mal3 sh10; do
    run_opt="-IF Player-Data/d" ./run-local.sh $i A prob 1 10 $opts
done
