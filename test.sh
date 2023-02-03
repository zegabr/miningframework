#!/bin/bash

./gradlew assemble

names=(matplotlib)
miningframework_path=$(pwd)
results_path="$miningframework_path/bug_results"

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
  rm -rf "$results_path"/${i}_results/
  ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -t 4 -s 29/01/2023 -u 31/01/2023 ./projects/${i}.csv "$results_path"/${i}_results"

  if [[ "$i" == "${names[0]}" ]]; then
    head -n 1 "$results_path"/${i}_results/results.csv > "$results_path"/all_results.csv
    head -n 1 "$results_path"/${i}_results/results.csv > "$results_path"/relevant.csv
  fi
    grep -e "true" -e "false" "$results_path"/${i}_results/results.csv >> "$results_path"/all_results.csv
  # clearing miningframework irrelevant cases (csdiff == diff3)
  cd "$results_path"/${i}_results/
  echo "deleting diff3 == csdiff folders"
  bash "$miningframework_path"/clear_irrelevant.sh

# populating relevant csv
if [ -d "$results_path"/${i}_results/${i}/ ]; then
  cd "$results_path"/${i}_results/${i}/
  echo "populating relevant.csv with only folders where csdiff != diff3"
  get_relevant_csv >> "$results_path"/relevant.csv
  cd "$miningframework_path"/
fi

# re_running csdiff v3 to see check for mistakes
if [ "$(ls -A "$results_path" | wc -l)" -gt 1 ]; then
    cd "$results_path"/${i}_results/
    echo "running csdiff again for the missing folders"
    bash "$miningframework_path"/re_run_csdiffv3.sh
fi
done




