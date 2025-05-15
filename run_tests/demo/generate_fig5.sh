#!/bin/bash

# create the following directories and files for storing original and refactored data
# dataset/
# ├── Hurricane/
# │   ├── data/
# │   |   ├── VelocityX.dat
# │   |   ├── VelocityY.dat
# │   |   └── VelocityZ.dat
# │   └── refactor/
# │       ├── VelocityX_refactored/
# |       |   └── metadata.bin
# │       ├── VelocityY_refactored/
# |       |   └── metadata.bin
# │       └── VelocityZ_refactored/
# |           └── metadata.bin
# └── NYX/
#     ├── data/
#     |   ├── VelocityX.dat
#     |   ├── VelocityY.dat
#     |   └── VelocityZ.dat
#     └── refactor/
#         ├── VelocityX_refactored/
#         |   └── metadata.bin
#         ├── VelocityY_refactored/
#         |   └── metadata.bin
#         └── VelocityZ_refactored/
#             └── metadata.bin
top_dirs=("dataset/Hurricane" "dataset/NYX")
sub_dirs=("data" "refactor/VelocityX_refactored" "refactor/VelocityY_refactored" "refactor/VelocityZ_refactored")
for top_dir in "${top_dirs[@]}"; do
    mkdir -p "$top_dir"
    for sub_dir in "${sub_dirs[@]}"; do
        mkdir -p "$top_dir/$sub_dir"
        if [[ $sub_dir == refactor/* ]]; then
            touch "$top_dir/$sub_dir/metadata.bin"
        fi
    done
done

# download data
sh ./run_tests/demo/download_data.sh


data=("NYX" "Hurricane")
for it in "${data[@]}"; do
    # T1: refactor
    ./build/test/refactor_data $it ./dataset/$it
    # T2: run tests
    mkdir -p ./run_tests/${it}_logs
    sh ./run_tests/${it}.sh ./dataset/$it

done

# T3: generate figures
mkdir -p ./plots/figures
/usr/bin/python3 ./plots/fig5.py