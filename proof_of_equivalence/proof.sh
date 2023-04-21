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
   run_csdiff awk_optimization
}

# ===== MAIN =====
# step1: run csdiff_v2 inside big_file_example directory
cd $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example
csdiff_v2
cd ..
cp -r $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example $MININGFRAMEWORK_DIR/proof_of_equivalence/csdiff_output_v2



# step 3: run csdiff awk inside big_file_example directory
cd $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example
csdiff_awk
cd ..
cp -r $MININGFRAMEWORK_DIR/proof_of_equivalence/big_file_example $MININGFRAMEWORK_DIR/proof_of_equivalence/csdiff_output_awk

# step 4: compare output not considering whitespaces
diff -r --ignore-all-space $MININGFRAMEWORK_DIR/proof_of_equivalence/csdiff_output_v2 $MININGFRAMEWORK_DIR/proof_of_equivalence/csdiff_output_awk
# step 5: print message if files are equal
if [ $? -eq 0 ]
then
    echo "Results are equal"
else
    echo "Results are not equal"
fi

