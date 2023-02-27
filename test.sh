#!/bin/bash

./gradlew assemble

names=(matplotlib tensorflow certbot flask ipython requests salt scrapy sentry tornado)
miningframework_path=$(pwd)
results_path="$miningframework_path/mining_results"

echo "deleting all_results"
rm "$results_path"/all_results.csv

get_relevant_csv(){
    # Store the output of ls -1 in an array
    arr=($(ls -1))

    # Start building the command
    command="grep"

    # Add each element in the array to the command
    for elem in "${arr[@]}"; do
        command="$command -e \"$elem\""
    done

    # Add the file path
    command="$command "$results_path"/all_results.csv"

    # Run the command
    eval $command
}


for i in "${names[@]}"
do
    cd "$miningframework_path"/
    echo "removing last results"
    rm -rf "$results_path"/${i}_results/

    echo "running mining framework"
    ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) , :' -t 1 -s 01/01/2021 -u 01/01/2022 ./projects/${i}.csv "$results_path"/${i}_results"

    if [[ -e "$results_path"/${i}_results/results.csv ]]; then
        if [[ ! -e "$results_path"/all_results.csv ]]; then
            echo "created all_results and relevant csvs"
            head -n 1 "$results_path"/${i}_results/results.csv > "$results_path"/all_results.csv
            head -n 1 "$results_path"/${i}_results/results.csv > "$results_path"/relevant.csv
        fi
        grep -e "true" -e "false" "$results_path"/${i}_results/results.csv >> "$results_path"/all_results.csv
    fi

    # re_running csdiff v3 to see check for mistakes
    # if [ "$(ls -A "$results_path" | wc -l)" -gt 1 ]; then
    #     cd "$results_path"/${i}_results/
    #     echo "running csdiff again for the missing folders"
    #     bash "$miningframework_path"/re_run_csdiffv3.sh
    # fi

    # clearing miningframework irrelevant cases (csdiff == diff3)
    if [ -d "$results_path"/${i}_results/ ]; then
        cd "$results_path"/${i}_results/
        echo "deleting diff3 == csdiff folders"
        bash "$miningframework_path"/clear_irrelevant.sh
    fi

    # populating relevant csv
    if [ -d "$results_path"/${i}_results/${i}/ ]; then
      cd "$results_path"/${i}_results/${i}/
      echo "populating relevant.csv with only folders where csdiff != diff3"
      get_relevant_csv >> "$results_path"/relevant.csv
    fi
done




