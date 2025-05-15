#!/bin/bash

run_tests_folder="./run_tests"

log_files=$(find "$run_tests_folder" -type f -name "*.log")

# output_file="elapsed_times.txt"
# > "$output_file"

total_elapsed_time=0

for log_file in $log_files; do
    elapsed_times=$(grep "elapsed_time = " "$log_file" | awk -F'=' '{print $2}' | xargs)
    # echo "$log_file: $elapsed_time" >> "$output_file"
    for elapsed_time in $elapsed_times; do
        total_elapsed_time=$(echo "$total_elapsed_time + $elapsed_time" | bc)
    done
done

# echo "Total elapsed time: $total_elapsed_time" >> "$output_file"
total_elapsed_time_minutes=$(echo "scale=10; $total_elapsed_time / 60" | bc)
echo "Total elapsed time: $total_elapsed_time_minutes min"
