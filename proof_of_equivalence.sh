#!/bin/bash

rm -rf ./v2_results
# TODO: change src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy csdiff manually to have "dependencies/csdiff_v2.sh" on line 10
sed -i 's/csdiff_awk_optimization.sh/csdiff_v2.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
echo "running mining framework"
./gradlew assemble
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) ,' -t 1 -s 01/01/2021 -u 03/01/2021 ./python_projects_backup.csv ./v2_results"

rm -rf ./awk_results
# use sed to change the "dependencies/csdiff_v2.sh" on file src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy to be "dependencies/csdiff_awk_optimization.sh"
sed -i 's/csdiff_v2.sh/csdiff_awk_optimization.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
./gradlew assemble
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) ,' -t 1 -s 01/01/2021 -u 03/01/2021 ./python_projects_backup.csv ./awk_results"

# get list of all files ending in csdiff.py for both results, sort them, and for each file awk_results, check for the difference in the same file of the v2_results
awk_results=$(find ./awk_results | grep csdiff.py)
v2_results=$(find ./v2_results | grep csdiff.py)
has_difference=0
for file in $awk_results; do
    echo "comparing $file"
    # if file exists in v2_results
    if [ -f $(echo $file | sed 's/awk_results/v2_results/g') ]; then
        diff -B $file $(echo $file | sed 's/awk_results/v2_results/g')
        if [ $? -ne 0 ]; then
            echo "files are different"
            # stop the loop
            break
            has_difference=1
        fi
    else 
        echo "file does not exist in v2_results"
        # stop the loop
        break
        has_difference=1
    fi

done

# since 0 is true, it is checking for false
if [ -z "$has_difference" ]; then
    echo "all files are the same"
fi

sed -i 's/csdiff_awk_optimization.sh/csdiff_v2.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
