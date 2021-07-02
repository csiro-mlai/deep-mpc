#!/bin/bash

if test -z "$5"; then
    echo "Usage: $0 <protocol> <net> <rounding> <n_threads> <n_epochs> <precision> [<further...>]"
    exit 1
fi

protocol=$1
net=$2
round=$3
n_threads=$4
n_epochs=$5
f=$6
shift 6

cd MP-SPDZ

case $protocol in
    sh3) protocol=ring ;;
    mal3) protocol=sy-rep-ring ;;
    mal4) protocol=rep4-ring ;;
    emul) protocol=emulate; compile_args=-K ;;
esac

args="$*"

if [[ $net = D ]]; then
    args="2dense $args"
fi

if [[ $protocol == rep4-ring ]]; then
    args="split4 $args"
elif [[ $protocol != emulate ]]; then
    args="split3 $args"
fi

k=$[2 * f - 1]
args="f$f k$k $args"

if [[ $round = near ]]; then
    args="nearest $args"
elif [[ $protocol != sy-rep-ring && $round = prob ]]; then
    args="trunc_pr $args"
fi

args="mnist_full_$net $n_epochs 128 $n_threads $args"

python3 ./compile.py $compile_args -CDR 64 $args | grep -v WARNING

touch ~/.rnd
Scripts/setup-ssl.sh 4

for i in $(seq 0 $[N-1]); do
    echo $i
    echo "${hosts[$i]}"
done

args=${args% }
prog=${args// /-}

bin=$protocol-party.x

if [[ $protocol = ring ]]; then
    bin=replicated-ring-party.x
elif [[ $protocol = emulate ]]; then
    bin=emulate.x
fi
