#!/bin/bash

qois=("VTOT" "T" "C" "Mach" "PT" "mu")

a=0.1
r=0.5
n=20
ebs=()
for ((i=1; i<=n; i++)); do
    ebs+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

data="GE"

mkdir -p run_tests/${data}_logs

data_prefix_path=$1

for it in "${qois[@]}"; do

    # PMGARD-HB
    log="./run_tests/${data}_logs/pmgard_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_$it $eb $data_prefix_path"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

    # PSZ3
    log="./run_tests/${data}_logs/psz3_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_${it}_sz3 $eb $data_prefix_path"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

    # PSZ3-delta
    log="./run_tests/${data}_logs/psz3delta_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_${it}_sz3delta $eb $data_prefix $rdata_prefix"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

done
