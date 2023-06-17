#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=6:00:00
#SBATCH -J frgbbvi
#SBATCH -o logs/frgbbvi.o%j

module purge
module load cuda cudnn gcc 
source activate tf210



D=$1
for D in $D 
do
    for B in 2  ;
    do
        echo 'batch : ' $B
        for lr in 0.1  ;
        do
            for seed in {0..10}
            do
                time python -u frgaussian_bbvi.py -D $D -r $D --batch $B --niter 50000 --lr $lr --seed $seed &
            done ;
            wait 
        done ;
    done
done
wait


# c=$1
# D=10
# for D in $D 
# do
#     for B in  2 ;
#     do
#         echo 'batch : ' $B
#         for lr in  0.01  0.1 0.001 ;
#         do
#             for seed in {0..10}
#             do
#                 time python -u illcondfrg_bbvi.py -D $D -c $c --batch $B --niter 20000 --lr $lr --seed $seed &
#             done
#             wait
#         done ;
#     done ;
# done
# wait
