#!/bin/bash

i=0
for host in $(<HOSTS); do
    hosts[$i]=$host
    i=$[i+1]
done

. common.sh

{
    for i in $(seq 0 $[N-1]); do
	echo $i ${hosts[$i]}
	{
	    ssh ${hosts[$i]} "
for i in static/*.x; do killall ${i#static/}; done
	"
	    rsync -Rltv static/$bin Programs/Bytecode/$prog-* Programs/Schedules/$prog.sch Player-Data/*.{pem,key,0} Player-Data/*Input-* ${hosts[$i]}:
	}&
    done
} 2>&1 | grep -v 'Permanently added' | grep -v 'no process found'

wait

setup_opts="-h ${hosts[0]} -pn $[RANDOM+1024] -p"
log=$(echo $prog-$(basename $bin) | sed 's/ //g')

test -e logs || mkdir logs

{
for j in $(seq 0 $[N-1]); do
    { while true; do echo; sleep 1; done; } |
    ssh ${hosts[$j]} "
c_rehash Player-Data
echo $prefix
$prefix static/$bin $prog $run_opt $setup_opts $j
	" 2>&1 | {
        logfile=logs/$log-$j
        echo logging to $logfile
        if true || test $j -eq 0; then
            date >> $logfile
            tee -a $logfile
            date >> $logfile
        else
            cat >> /dev/null
        fi
    } & true
done
} 2>&1 | grep -v 'Permanently added' | grep -v 'no process found'

# allow debugging with docker
exit 0
