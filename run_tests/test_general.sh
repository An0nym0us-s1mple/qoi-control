#!/bin/bash

usage() {
    echo "Usage: $0 <dataset> <data_prefix_path>"
    echo "dataset should be one of the following: Hurricane, NYX, GE, or S3D"
    echo "data_prefix_path/data/ should contain the original data files before execution"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage
fi

data=$1
data_prefix_path=$2

# check for valid dataset name
if [[ "$data" != "Hurricane" && "$data" != "NYX" && "$data" != "GE" && "$data" != "S3D" ]]; then
    echo "Invalid dataset name: $data"
    usage
fi


# T1: refactor data
if ! ./build/test/refactor_data "$data" "$data_prefix_path"; then
    echo "Error in refactoring data"
    exit 1
fi

# T2: run tests
mkdir -p "./run_tests/${data}_logs"
if ! sh "./run_tests/${data}.sh" "$data_prefix_path"; then
    echo "Error in running tests for $data"
    exit 1
fi

# T3: generate figures
mkdir -p ./plots/figures
case "$data" in
    "NYX")
        /usr/bin/python3 ./plots/fig5_1.py
        ;;
    "Hurricane")
        /usr/bin/python3 ./plots/fig5_2.py
        ;;
    "GE")
        /usr/bin/python3 ./plots/fig4.py
        /usr/bin/python3 ./plots/fig7.py
        ;;
    "S3D")
        /usr/bin/python3 ./plots/fig6.py
        /usr/bin/python3 ./plots/fig8.py
        ;;
esac

echo "All tests for $data executed successfully."
