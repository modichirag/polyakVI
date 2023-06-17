#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=6:00:00
#SBATCH -J ngbbvi
#SBATCH -o logs/ngbbvi.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


D=10
B=2

t=1.0
for lr in  0.1  ;
do
    for B in 2  ;
    do
        for s in  $(seq 0.0 .2 2.0) ;
        do
            echo $s
            for seed in {0..10}
            do
                time python -u nongaussian_bbvi.py -D $D --batch $B --niter 10000 --skewness $s --tailw $t --seed $seed & 
            done ;
            wait
        done ;
    done
done
wait
    

# s=0.0
# for lr in 0.1 ;
# do
#     for B in 2  ;
#     do
#         for t in  $(seq 0.1 .2 2.0) ;
#         do
#             echo $t
#             for seed in {0..10}
#             do
#                 time python -u nongaussian_bbvi.py -D $D --batch $B --niter 10000 --skewness $s --tailw $t --seed $seed &
#             done
#             wait
#         done ;
#     done ;
# done
# wait

