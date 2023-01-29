#!/bin/bash

./gradlew assemble

names=(matplotlib tensorflow certbot flask ipython requests salt scrapy sentry tornado)

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
    command="$command /home/ze/Desktop/mining_results/all_results.csv"

    # Run the command
    eval $command
}

for i in "${names[@]}"
do
  rm -rf /home/ze/Desktop/mining_results/${i}_results/
  ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/${i}.csv /home/ze/Desktop/mining_results/${i}_results"

  if [[ "$i" == "${names[0]}" ]]; then
    head -n 1 /home/ze/Desktop/mining_results/${i}_results/results.csv > /home/ze/Desktop/mining_results/relevant.csv
    head -n 1 /home/ze/Desktop/mining_results/${i}_results/results.csv > /home/ze/Desktop/mining_results/all_results.csv
  fi
    grep -e "false" /home/ze/Desktop/mining_results/${i}_results/results.csv  >> /home/ze/Desktop/mining_results/relevant.csv
    grep -e "true" -e "false" /home/ze/Desktop/mining_results/${i}_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
  # clearing miningframework mistakes
  cd /home/ze/Desktop/mining_results/${i}_results/
  bash /home/ze/miningframework/clear_irrelevant.sh
  # populating relevant csv
  cd /home/ze/Desktop/mining_results/${i}_results/${i}/
  get_relevant_csv > /home/ze/Desktop/mining_results/relevant.csv
  cd /home/ze/miningframework/
done




