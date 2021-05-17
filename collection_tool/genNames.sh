#!/bin/bash

# A script to automatically generate mp3s for each voice with generate_audio_queries.py

if [ $# -ne 1 ] || [ ! -f $1 ]; then
    echo "Provide csv file with query/times"
    exit
fi
    

declare -a names=("aditi" "amy" "brian" "emma" "ivy" "joanna" "joey" "justin" "kendra" "kevin" "kimberly" "matt" "nicole" "olivia" "raveena" "russell" "salli")

for name in "${names[@]}"
do
    ./generate_audio_queries.py --voice "$name" $1
done
