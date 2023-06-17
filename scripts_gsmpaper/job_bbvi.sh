#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=5:00:00
#SBATCH -J bbvi
#SBATCH -o logs/bbvi.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


nmodel=$1
echo "For model number : " $nmodel

lrmap=0.01
nclimb=10000
niter=50000
mode='full'
suffix=''
modeinit=1
scale=1.0


for lr in 0.01 0.00l 0.1 ;
do
    for batch in 2   ;
    do
        for seed in {0..10}
        do
            echo "running model " $nmodel " with seed " $seed " and batch " $batch
            time python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map $lrmap --nclimb $nclimb --niter $niter --batch $batch --lr $lr --mode $mode --modeinit $modeinit  --scale $scale --seed $seed  &
        done
        wait
    done
    wait
done
wait 

# for batch in 32 64 128 ;
# #for batch in 1 2 4 8 16 32 ;
# do
#     python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 50000 --batch $batch --lr 0.00001 --mode full --modeinit 2 --suffix _meaninit &
#     python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 50000 --batch $batch --lr 0.0001 --mode full --modeinit 2 --suffix _meaninit &
#     python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 50000 --batch $batch --lr 0.001 --mode full --modeinit 2 --suffix _meaninit &
#     python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 50000 --batch $batch --lr 0.01 --mode full --modeinit 2 --suffix _meaninit  &
#     python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 50000 --batch $batch --lr 0.1 --mode full --modeinit 2 --suffix _meaninit  &
# '    echo "running for single batch"
#     wait 
# done
# wait


# # for nmodel in 0 44 64 68 69 31 85 ;
# do
#     for batch in 1 2 4 8 16 32 ;
#     do
#         python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 20000 --batch $batch --lr 0.001 &
#         python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 20000 --batch $batch --lr 0.01 &
#         python  -u fitq_pdb_bbvi.py --nmodel $nmodel --lr_map 0.01 --nclimb 1000 --niter 20000 --batch $batch --lr 0.1 &
#         echo "running for single batch"
#         wait 
#     done
#     wait
# done
# wait
