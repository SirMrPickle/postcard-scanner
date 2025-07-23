#!/bin/bash

input_dir="_RAW"
start_id=146  # Start numbering here

if [[ ! -d "$input_dir" ]]; then
  echo "Error: directory '$input_dir' not found."
  exit 1
fi

# Get list of files matching pattern, sort numerically by suffix NOT lexicographically, but by SUFFIX. Much better
files=($(ls "$input_dir" | grep '^SimpleScanStation[0-9]\{14\}_[0-9]\+\.png$' | sort -t '_' -k2 -n))

count=$start_id
total=${#files[@]}

if (( total % 2 != 0 )); then
  echo "Warning: Odd number of files found. Last one will be skipped."
  total=$((total - 1))
fi

for ((i=0; i<total; i+=2)); do
  num=$(printf "%02d" $count)
  front="${files[$i]}"
  back="${files[$i+1]}"

  mv "$input_dir/$front" "$input_dir/sc${num}-front.png"
  mv "$input_dir/$back" "$input_dir/sc${num}-back.png"

  ((count++))
done

echo "Renaming done: $((count - start_id)) pairs renamed, starting from ID $start_id."
