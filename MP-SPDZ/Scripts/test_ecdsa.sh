#!/usr/bin/env bash

echo SECURE = -DINSECURE >> CONFIG.mine
touch ECDSA/Fake-ECDSA.cpp

make -j4 ecdsa Fake-ECDSA.x

run()
{
    echo $1
    port=$[RANDOM+1024]
    if ! {
	    for j in $(seq 0 $2); do
		./$1-ecdsa-party.x -pn $port -p $j 1 2>/dev/null & true
	    done
	    wait
	} | grep "Online checking"; then
	exit 1
    fi
}

for i in rep mal-rep shamir mal-shamir atlas sy-rep; do
    run $i 2
done

run rep4 3

for i in semi mascot; do
    run $i 1
done

./Fake-ECDSA.x
run fake-spdz 1
