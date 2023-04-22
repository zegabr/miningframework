#!/bin/bash

# Run this inside proof_of_equivalence folder

MININGFRAMEWORK_DIR="$(dirname "$(pwd)")/"

run_csdiff(){
    bash $MININGFRAMEWORK_DIR/dependencies/csdiff_$1.sh -s "( ) , :" ./left.py ./base.py ./right.py
}

csdiff_v2 ()
{
   run_csdiff v2
}

csdiff_awk ()
{
   run_csdiff awk_simplification
}

NUMBER_OF_TIMES=10
run_v2_many_times(){
    echo "runing v2"
   for i in $(seq 1 $NUMBER_OF_TIMES)
    do
        cd $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example
        csdiff_v2
        cd ..
    done
}

run_awk_many_times(){
    echo "runing awk"
   for i in $(seq 1 $NUMBER_OF_TIMES)
    do
        cd $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example
        csdiff_awk
        cd ..
    done
}
# ===== MAIN =====

time run_v2_many_times
time run_awk_many_times
