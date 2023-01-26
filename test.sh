#!/bin/bash

./gradlew assemble

rm -rf /home/ze/Desktop/mining_results/matplotlib_results/
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/matplotlib.csv /home/ze/Desktop/mining_results/matplotlib_results"

# rm -rf /home/ze/Desktop/mining_results/tensorflow_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/09/2021 -u 01/01/2022 ./projects/tensorflow.csv /home/ze/Desktop/mining_results/tensorflow_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/certbot_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/11/2021 -u 01/01/2022 ./projects/certbot.csv /home/ze/Desktop/mining_results/certbot_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/flask_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/flask.csv /home/ze/Desktop/mining_results/flask_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/ipython_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/ipython.csv /home/ze/Desktop/mining_results/ipython_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/requests_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/requests.csv /home/ze/Desktop/mining_results/requests_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/salt_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/salt.csv /home/ze/Desktop/mining_results/salt_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/scrapy_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/scrapy.csv /home/ze/Desktop/mining_results/scrapy_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/sentry_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/sentry.csv /home/ze/Desktop/mining_results/sentry_results"

# cd /home/ze/Desktop/mining_results/
# bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
# cd -

# rm -rf /home/ze/Desktop/mining_results/tornado_results/
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/01/2022 ./projects/tornado.csv /home/ze/Desktop/mining_results/tornado_results"

# fetting relevant results
head -n 1 /home/ze/Desktop/mining_results/matplotlib_results/results.csv > /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/matplotlib_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/tensorflow_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/certbot_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/flask_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/ipython_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/requests_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/salt_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/scrapy_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/sentry_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv
cat /home/ze/Desktop/mining_results/tornado_results/results.csv | grep false >> /home/ze/Desktop/mining_results/relevant.csv

# getting final results
cat /home/ze/Desktop/mining_results/matplotlib_results/results.csv > /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/tensorflow_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/certbot_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/flask_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/ipython_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/requests_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/salt_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/scrapy_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/sentry_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
cat /home/ze/Desktop/mining_results/tornado_results/results.csv >> /home/ze/Desktop/mining_results/all_results.csv
# numberOfRelevantCommits=0
# numberOfRelevantCommits=$(cat /home/ze/Desktop/mining_results/relevant.csv | wc -l )
# numberOfRelevantCommits-=1

# echo $numberOfCommits
# echo $numberOfRelevantCommits

cd /home/ze/Desktop/mining_results/
bash /home/ze/Desktop/miningframework/clear_irrelevant.sh
cd -
