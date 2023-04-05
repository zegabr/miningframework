#!/bin/bash

cd ~/miningframework
# TODO: change src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy csdiff manually to have "dependencies/csdiff_v2.sh" on line 10
sed -i 's/csdiff_awk_optimization.sh/csdiff_v2.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
echo "running mining framework"
./gradlew assemble
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) ,' -t 1 -s 01/01/2021 -u 05/01/2021 ./python_projects_backup.csv ./v2_results"
cd ./v2_results && bash ~/miningframework/clear_irrelevant.sh

cd ~/miningframework
# use sed to change the "dependencies/csdiff_v2.sh" on file src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy to be "dependencies/csdiff_awk_optimization.sh"
sed -i 's/csdiff_v2.sh/csdiff_awk_optimization.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
./gradlew assemble
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) ,' -t 1 -s 01/01/2021 -u 05/01/2021 ./python_projects_backup.csv ./awk_results"
cd ./awk_results && bash ~/miningframework/clear_irrelevant.sh

cd ~/miningframework
# get list of all files ending in csdiff.py for both results, sort them, and for each file awk_results, check for the difference in the same file of the v2_results
awk_results=$(find ./awk_results -name "*csdiff.py" | sort)
v2_results=$(find ./v2_results -name "*csdiff.py" | sort)
has_difference=0
for file in $awk_results; do
    echo "comparing $file"
    diff -B $file $(echo $file | sed 's/awk_results/v2_results/g')
    if [ $? -ne 0 ]; then
        echo "files are different"
        # stop the loop
        break
        has_difference=1
    fi
done

# if has different number of files, set has_difference to 1
if [ $(echo $awk_results | wc -w) -ne $(echo $v2_results | wc -w) ]; then
    echo "different number of files"
    has_difference=1
fi

# since 0 is true, it is checking for false
if [ -z "$has_difference" ]; then
    echo "all files are the same"
fi

sed -i 's/csdiff_awk_optimization.sh/csdiff_v2.sh/g' src/main/services/dataCollectors/csDiffCollector/CSDiffRunner.groovy
