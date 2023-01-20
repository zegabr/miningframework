./gradlew assemble
# TODO: replicate block below to create one folder for each repo, so that way we can run each repo in a isolated maner by commenting this code
#
rm -rf ~/Desktop/tensorflow_results/
echo "results removed"
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./tensorflow.csv ~/Desktop/tensorflow_results"
cat ~/Desktop/tensorflow_results/results.csv | grep false
