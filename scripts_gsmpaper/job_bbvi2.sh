#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=12:00:00
#SBATCH -J bbvi2
#SBATCH -o logs/bbvi2.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


n1=$1
n2=$2
echo "For model number : " $n1 $n2

lrmap=0.01
nclimb=1000
niter=50000
mode='full'
suffix=''
modeinit=1
scale=1.0

for n in  $(seq  $n1 $n2)
do
    echo $n
    for lr in 0.01 0.001 0.1 ;
    do
        for batch in 2  ;
        do
            for seed in {0..1}
            do
                echo "running model " $n " with seed " $seed " and batch " $batch
                time python  -u fitq_pdb_bbvi.py --nmodel $n --lr_map $lrmap --nclimb $nclimb --niter $niter --batch $batch --lr $lr --mode $mode --modeinit $modeinit  --scale $scale --seed $seed --err 1 --log 1  &
            done
        done
    done
    wait 
done
