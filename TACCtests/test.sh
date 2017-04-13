# /bin/bash

runs=10
command="mpirun ../libcirc/mpibackend"

# load gsl module
module gsl

echo "HT Stack, 9 T gates, 10000 samples, $runs runs" > out.txt
date >> out.txt
for file in "HT9-000000000.test" "HT9-000000001.test" "HT9-000000010.test" "HT9-000000011.test"
do
    echo "" >> out.txt
    echo "----------------------------" >> out.txt
    echo $file >> out.txt
    for i in $(seq 1 1 $runs)
    do
        echo "" >> out.txt
        echo "---------" >> out.txt
        echo "Run $i" >> out.txt
        (time $command $file) &>> out.txt
    done
done


