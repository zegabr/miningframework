./gradlew assemble
rm -rf ~/Desktop/python_results/
echo "results removed"
./gradlew run --args="-e .py -i injectors.CSDiffModule -l '( ) : ,' -s 01/01/2021 -u 01/11/2021 ./python_projects.csv ~/Desktop/python_results"
cat ~/Desktop/python_results/results.csv | grep false
