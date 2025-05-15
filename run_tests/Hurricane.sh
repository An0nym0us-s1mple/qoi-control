#!/bin/bash

a=0.1
r=0.1
n=7
ebs=()
for ((i=1; i<=n; i++)); do
    ebs+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

data="Hurricane"

mkdir -p run_tests/${data}_logs

data_prefix_path=$1

log="./run_tests/${data}_logs/pmgard_VTOT_${data}.log"
touch -f $log

for eb in "${ebs[@]}"; do
    cmd="./build/test/halfing_Vtot_general $eb $data_prefix_path"
    echo "$cmd" >> "$log"
    eval "$cmd" >> "$log"
done