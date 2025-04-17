#!/bin/bash

declare -i u
declare -i i

for (( i=10; i>1; --i )); 
do
    u=$(( 10 * $i ));
    echo "Running with u = $u";
    py main.py $u;
done

