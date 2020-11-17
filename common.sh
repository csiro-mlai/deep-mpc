#!/bin/bash

if test -z "$1" -o -z "$2" -o -z "$3" -o -z "$4" -o -z "$5" -o -z "$6"; then
    echo "Usage: $0 <protocol> <n_layers> <round> <back_prop> <n_threads> <n_epochs> [no_loss]"
    exit 1
fi

protocol=$1
n_layers=$2
round=$3
back_prop=$4
n_threads=$5
n_epochs=$6
shift 6

cd MP-SPDZ

case $protocol in
    sh3) protocol=ring ;;
    mal3) protocol=sy-rep-ring ;;
    mal4) protocol=rep4-ring ;;
    emul) protocol=emulate ;;
esac

args="always_acc $*"

if [[ $n_layers < 3 ]]; then
    args="${n_layers}dense $args"
fi

if [[ $protocol == rep4-ring ]]; then
    args="split4 $args"
elif [[ $protocol != emulate ]]; then
    args="split3 $args"
fi

if [[ $round = near ]]; then
    args="nearest $args"
elif [[ $protocol != sy-rep-ring && $round = prob ]]; then
    args="trunc_pr $args"
fi

if [[ $back_prop = relu_prob ]]; then
    args="approx $args"
elif [[ $back_prop = relu_grad ]]; then
    args="relu_out $args"
fi

args="mnist_full_A $n_epochs 128 $n_threads $args"

python3 ./compile.py -CDR 64 $args | grep -v WARNING

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
