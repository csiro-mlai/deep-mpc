#!/bin/bash

if test -z "$6"; then
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

test -e logs || mkdir logs

case $protocol in
    sh2) protocol=hemi; PLAYERS=2; run_opt="-b 100 $run_opt" ;;
    sh3) protocol=ring; PLAYERS=3 ;;
    dm3) protocol=temi; PLAYERS=3; run_opt="-b 100 -lgp 111 $run_opt" ;;
    mal3) protocol=sy-rep-ring; PLAYERS=3 ;;
    mal4) protocol=rep4-ring; PLAYERS=4 ;;
    sh10) protocol=atlas; PLAYERS=10 ;;
    dm10) protocol=temi; PLAYERS=10; run_opt="-b 100 -lgp 111 $run_opt" ;;
    emul) protocol=emulate; compile_args="-K ''" ;;
esac

export PLAYERS

args="$*"

if [[ $net = D ]]; then
    args="2dense $args"
fi

if [[ $protocol =~ ring || $protocol == emulate ]]; then
    ring=1
    compile_args="-R 64 $compile_args"
    if [[ $protocol == rep4-ring ]]; then
	args="split4 $args"
    elif [[ $protocol != emulate ]]; then
	args="split3 $args"
    fi
else
    ring=0
    args="edabit $args"
fi

k=$[2 * f - 1]
args="f$f k$k $args"

if [[ $round = near ]]; then
    args="nearest $args"
elif [[ $protocol != sy-rep-ring && $round = prob &&
	    ($ring = 1 || $protocol = hemi) ]]; then
    args="trunc_pr $args"
fi

if [[ $net = alex ]]; then
    args="falcon_alex $n_epochs 128 $n_threads $args"
    run_opt="-IF /tmp/cifar10-Input $run_opt"
elif [[ $net = new_alex ]]; then
    args="alex $n_epochs 128 $n_threads $args"
    run_opt="-IF /tmp/cifar10-Input $run_opt"
else
    args="mnist_full_$net $n_epochs 128 $n_threads $args"
fi

python3 ./compile.py $compile_args -CD $args | grep -v WARNING

touch ~/.rnd
Scripts/setup-ssl.sh 10

N=$PLAYERS

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
