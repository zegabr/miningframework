./gradlew assemble
#
rm -rf /home/ze/Desktop/mining_results/matplotlib_results/
echo "results removed"
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/matplotlib.csv /home/ze/Desktop/mining_results/matplotlib_results"
cat /home/ze/Desktop/mining_results/matplotlib_results/results.csv | grep false

rm -rf /home/ze/Desktop/mining_results/tensorflow_results/
echo "results removed"
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/tensorflow.csv /home/ze/Desktop/mining_results/tensorflow_results"
cat /home/ze/Desktop/mining_results/tensorflow_results/results.csv | grep false

rm -rf /home/ze/Desktop/mining_results/ansible_results/
echo "results removed"
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/ansible.csv /home/ze/Desktop/mining_results/ansible_results"
cat /home/ze/Desktop/mining_results/ansible_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/certbot_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/certbot.csv /home/ze/Desktop/mining_results/certbot_results"
# cat /home/ze/Desktop/mining_results/certbot_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/compose_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/compose.csv /home/ze/Desktop/mining_results/compose_results"
# cat /home/ze/Desktop/mining_results/compose_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/django_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/django.csv /home/ze/Desktop/mining_results/django_results"
# cat /home/ze/Desktop/mining_results/django_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/flask_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/flask.csv /home/ze/Desktop/mining_results/flask_results"
# cat /home/ze/Desktop/mining_results/flask_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/ipython_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/ipython.csv /home/ze/Desktop/mining_results/ipython_results"
# cat /home/ze/Desktop/mining_results/ipython_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/pyspider_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/pyspider.csv /home/ze/Desktop/mining_results/pyspider_results"
# cat /home/ze/Desktop/mining_results/pyspider_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/requests_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/requests.csv /home/ze/Desktop/mining_results/requests_results"
# cat /home/ze/Desktop/mining_results/requests_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/salt_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/salt.csv /home/ze/Desktop/mining_results/salt_results"
# cat /home/ze/Desktop/mining_results/salt_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/scikit-learn_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/scikit-learn.csv /home/ze/Desktop/mining_results/scikit-learn_results"
# cat /home/ze/Desktop/mining_results/scikit-learn_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/scrapy_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/scrapy.csv /home/ze/Desktop/mining_results/scrapy_results"
# cat /home/ze/Desktop/mining_results/scrapy_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/sentry_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/sentry.csv /home/ze/Desktop/mining_results/sentry_results"
# cat /home/ze/Desktop/mining_results/sentry_results/results.csv | grep false

# rm -rf /home/ze/Desktop/mining_results/tornado_results/
# echo "results removed"
# ./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./projects/tornado.csv /home/ze/Desktop/mining_results/tornado_results"
# cat /home/ze/Desktop/mining_results/tornado_results/results.csv | grep false
