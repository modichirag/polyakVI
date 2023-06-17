#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=8:00:00
##SBATCH -J bbvipath
#SBATCH -o logs/%x.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


#pathfinder = 0 23 30 40 52 81 14 64 68 3 48 49 31 85 20 51 11 69 33 44 99
#glm = 0 23 30 40 52 81
#gp = 14 64 68
#gaussmix = 3 48 49
#heirarchical = 31 85
#diffeq = 20 51
#hmm = 11 69
#time series model = 33 44 99
#toshow = 23 68 48 31 51 11 44

lrmap=0.01
nclimb=10000
niter=50000
mode='full'
suffix=''
modeinit=1
scale=1.0


n1=$1
n2=$((n1+1))
echo "For model number : " $n1 $n2

for nmodel in $(seq $n1 $n2)
do 
    for lr in  0.01 ;
    do
        for batch in 2   ;
        do
            for seed in {0..10}
            do
                echo "running model " $nmodel " with seed " $seed " and batch " $batch
                time python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map $lrmap --nclimb $nclimb --niter $niter --batch $batch --lr $lr --mode $mode --modeinit $modeinit  --scale $scale --seed $seed --err 1 --log 1  &
            done
            wait
        done
        wait
    done
    wait
done

