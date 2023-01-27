#!/bin/bash

./gradlew assemble

names=(matplolib tensorflow certbot flask ipython requests salt scrapy sentry tornado)

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
  cd /home/ze/Desktop/mining_results/
  bash /home/ze/miningframework/clear_irrelevant.sh
  cd -
  echo "now you must remove all entries of relevant.csv that does not appear in the data (because those were mistakenly added to the result by small memory bugs)"
done
