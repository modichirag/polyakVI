#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=4:00:00
#SBATCH -J gsm
#SBATCH -o logs/gsm.o%j

module purge
module load cuda cudnn gcc 
source activate tf210

n1=$1
n2=$2
echo "For model number : " $n1 $n2

modeinit=1
scale=1.0

for nmodel in  $(seq  $n1 $n2)
do
    echo $nmodel
    for batch in 2   ;
    do
        for seed in {0..10}
        do 
            echo "running model " $nmodel " with seed " $seed " and batch " $batch
            python  -u fitq_pdb_gsm.py --nmodel $nmodel --lr_map 0.01 --nclimb 10000 --niter 10000 --batch $batch --modeinit $modeinit --seed $seed --scale $scale --err 1 --log 1  &
        done
        wait 
    done
    wait
done
wait


# for nmodel in 0 44 64 68 69 31 85 51 901 902 903 ;
# do
#     for batch in 1 2 4 8 16 ;
#     do
#         python  -u fitq_pdb_gsm.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 5000 --batch $batch --modeinit 1 &
#         echo "running for single batch with random init"
#     done
#     wait
# done
# wait

##for nmodel in 0 44 64 68 69 31 85 ;
##do
##    for batch in 1 2 4 8 16 ;
##    do
##        python  -u fitq_pdb_gsm.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 2000 --batch $batch --modeinit 1 --scale 0.1 &
##        echo "running for single batch with mode init, scale=0.1"
##    done
##    wait
##done
##wait
##
##for nmodel in 0 44 64 68 69 31 85 ;
##do
##    for batch in 1 2 4 8 16 ;
##    do
##        python  -u fitq_pdb_gsm.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 2000 --batch $batch --modeinit 0 --scale 0.1 &
##        echo "running for single batch with random init, scale=0.1"
##    done
##    wait
##done
##wait
##
