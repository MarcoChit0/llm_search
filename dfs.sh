#!/bin/bash
for budget in 4 16 64 256
do
    for value in "weak" "medium" "strong" "none"
    do
        python run.py -bis=900 -bie=1000 -cm=Qwen2.5-3B-Instruct -cs=dfs  -sl=$value -b=$budget
    done
done