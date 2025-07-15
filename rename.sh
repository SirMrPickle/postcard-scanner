#!/bin/bash

input_dir="_INPUT"
start_id=60  # Change this to whatever number you want to start from

if [[ ! -d "$input_dir" ]]; then
  echo "Error: directory '$input_dir' not found."
  exit 1
fi

files=()
while IFS= read -r line; do
  files+=("$line")
done < <(ls "$input_dir" | grep '^SimpleScanStation[0-9]\{14\}_[0-9]\+\.png$' | sort)

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