#!/bin/bash

# Number of times the script will be executed.
n=${1:-10}

rm -f time.txt
rm -f out.txt
rm -f outConsole.txt
rm -f output/data/soot-results.csv
rm -f output/data/results.pdf
rm -r output/results
mkdir -p output/results

# Loop to execute the script gradlew run n times
for ((i=1;i<=$n;i++))
do
    folder_name="execution-$i"
    mkdir $folder_name

    # Execute the script gradlew run
    ./gradlew run -DmainClass="services.outputProcessors.soot.Main" --args="-icf -ioa -idfp -pdg -report -t 0"

    # Move the files generated by the script to the current execution folder
    mv outConsole.txt time.txt output/data/soot-results.csv output/data/results.pdf $folder_name
done

mv -f execution-* output/results
find . -name "results_*" -type f -delete
find . -name "resultTime*" -type f -delete

python3 scripts/generate_time_csv_from_logs.py $n
python3 scripts/summarize_time_results.py $n

mv results_by_analysis.jpg output/results
mv results_by_scenarios.jpg output/results
mv results_analysis.pdf output/results
mv results_scenarios.pdf output/results
mv results_by_execution.jpg output/results
mv results_execution.pdf output/results

mkdir -p output/results/times
mv resultTime* output/results/times

python3 scripts/check_diff_results_pdf.py $n
mv diff_files.pdf output/results/
