#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J frg
#SBATCH -o logs/frg.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


# r=$1
# D=1000
# for B in  4 ;
# do
#     echo $B
#     for r in $r ;
#     do
#         echo 'rank : ' $r
#         for qs in 0 1 ;
#         do
#             time python -u frgaussian.py -D $D -r $r --batch $B --niter 10000 --qsample $qs --scale 0.01 & 
#         done ;
#         wait 
#     done ;
#     wait 
# done
# wait




D=$1
qs=1
for D in $D 
do
    for B in  2  ;
    do
        for seed in {0..10}
        do
            time python -u frgaussian.py -D $D -r $D --batch $B --niter 5000 --qsample $qs --seed $seed &
        done
        wait
    done ;
done
wait


# c=$1
# D=10
# qs=1

# for B in  2  ;
# do
#     for D in $D 
#     do
#         for c in 1 10 100 1000 ;
#         do
#             for seed in {0..10}
#             do
#                 time python -u illcondfrg.py -D $D -c $c --batch $B --niter 5000 --qsample $qs --seed $seed &
#             done
#             wait
#         done ;
#     done
# done
# wait
