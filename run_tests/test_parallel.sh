#!/bin/bash

# assume the original bp files of 96 blocks are stored in dataset/GE-large/data/sol_4114800_aver_b*.bp/
# create the following directories and files for storing refactored data
# dataset/
# └── GE-large/
#     ├── retrieved/
#     ├── data/
#     |   ├── sol_4114800_aver_b0.bp/
#     |   |   ├── data.0
#     |   |   ├── md.0
#     |   |   └── md.idx
#     |   ├── ...
#     |   └── sol_4114800_aver_b95.bp/
#     |       ├── data.0
#     |       ├── md.0
#     |       └── md.idx
#     └── refactor/
#         ├── block_0_refactored/
#         |   ├── VelocityX/
#         |   |   └── metadata.bin
#         |   ├── VelocityY/
#         |   |   └── metadata.bin
#         |   └── VelocityZ/
#         |       └── metadata.bin
#         ├── ...
#         └── block_95_refactored/
#             ├── VelocityX/
#             |   └── metadata.bin
#             ├── VelocityY/
#             |   └── metadata.bin
#             └── VelocityZ/
#                 └── metadata.bin
top_dir="dataset/GE-large"
mkdir -p "$top_dir"
mkdir -p "$top_dir/retrieved"
vars=("VelocityX", "VelocityY", "VelocityZ")
for i in {0..95}; do
    sub_dir="refactor/block_${i}_refactored"
    mkdir -p "$$top_dir/$sub_dir"
    for var in "${vars[@]}"; do
        mkdir -p "$$top_dir/$sub_dir/$var"
        touch -f "$$top_dir/$sub_dir/$var/metadata.bin"
    done
done

flag=0

# T1: refactor data
for cmd in refactorGE refactorGE_SZ3 refactorGE_SZ3_delta; do
    if ! ./build/parallel_src/$cmd ./dataset/GE-large; then
        echo "Error in refactoring data using $cmd"
        flag=1
    fi
done
(( flag == 1 )) && exit 1

# T2: run tests
mkdir -p ./run_tests/GE-large_logs
for cmd in testGE_VTOT testGE_VTOT_SZ3 testGE_VTOT_SZ3_delta; do
    mkdir -p "./run_tests/GE-large_logs/$cmd"
    # touch "$log"
    for i in {1..5}; do
        log="./run_tests/GE-large_logs/$cmd/1e-$i.log"
        test_cmd="mpirun -np 96 ./build/parallel_src/$cmd 1e-$i ./dataset/GE-large"
        echo "$test_cmd" >> "$log"
        if ! "$test_cmd" >> "$log"; then
            echo "Error in running $test_cmd"
            flag=1
        fi
    done
done
(( flag == 1 )) && exit 1

# T3: transfer data using Globus
# instructions for T3 is omitted because it cannot be completed offline