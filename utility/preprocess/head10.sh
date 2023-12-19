#!/bin/bash

# Check if a file name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Get the input file from the first argument
input_file=$1

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "File not found: $input_file"
    exit 1
fi

# Loop through the first 10 lines of the file
for i in $(seq 1 10)
do
    # Read the line
    line=$(sed -n ${i}p $input_file)

    # Check if the line is not empty
    if [ ! -z "$line" ]
    then
        # Create a new JSON file for each line
        echo $line | jq . > "nq_dev_line_${i}.json"
    fi
done

