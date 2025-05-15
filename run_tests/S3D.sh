#!/bin/bash


a=0.1
r=0.5
n=20
ebs=()
for ((i=1; i<=n; i++)); do
    ebs+=($a)
    a=$(echo "scale=10; $a * $r" | bc)
done

data="S3D"

mkdir -p run_tests/${data}_logs

data_prefix_path=$1

qois=("x1x3" "x4x5" "x0x4" "x3x5")

xis=(1 4 0 3)
xjs=(3 5 4 5)


for i in {0..3}; do
    it="${qois[i]}"

    # PMGARD-HB
    log="./run_tests/${data}_logs/pmgard_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_xixj ${xis[i]} ${xjs[i]} $eb $data_prefix_path"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

    # PSZ3
    log="./run_tests/${data}_logs/psz3_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_xixj_sz3 ${xis[i]} ${xjs[i]} $eb $data_prefix_path"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

    # PSZ3-delta
    log="./run_tests/${data}_logs/psz3delta_$it.log"
    touch -f $log

    for eb in "${ebs[@]}"; do
        cmd="./build/test/halfing_xixj_sz3delta ${xis[i]} ${xjs[i]} $eb $data_prefix_path"
        echo "$cmd" >> "$log"
        eval "$cmd" >> "$log"
    done

done
