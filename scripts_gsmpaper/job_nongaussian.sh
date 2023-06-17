#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J ngauss
#SBATCH -o logs/ngauss.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


D=10
B=2

t=1.0
for s in  $(seq 0.0 .2 2.0) ;
do
    echo $s
    for qs in 1 ;
    do
        for seed in {0..10}
        do
            time python -u nongaussian.py -D $D --batch $B --niter 5000 --qsample $qs --skewness $s --tailw $t --seed $seed & 
        done ;
        wait
    done
done ;
wait 


# s=0.0
# for t in  $(seq 0.1 .2 2.0) ;
# do
#     echo $t
#     for qs in  1 ;
#     do
#         for seed in {0..10}
#         do
#             time python -u nongaussian.py -D $D --batch $B --niter 5000 --qsample $qs --skewness $s --tailw $t --seed $seed & 
#         done ;
#         wait
#     done
# done ;
# wait 


