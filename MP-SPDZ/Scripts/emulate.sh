#!/bin/bash

dir="$(dirname $0)"
. "$dir"/run-common.sh
prog=${1%.sch}
prog=${prog##*/}
shift
mkdir logs 2> /dev/null
$prefix "$dir"/../emulate.x $prog $* 2>&1 | tee logs/emulate-$prog
