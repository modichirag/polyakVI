#!/bin/bash
#SBATCH -p ccm
#SBATCH --nodes=1
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH -J mode
#SBATCH -o logs/mode.o%j

module purge
module load cuda cudnn gcc 
source activate tf210


n1=$1
n2=$2
echo "For model number : " $n1 $n2


nclimb=1000

for nmodel in  $(seq  $n1 $n2)
do
    echo $nmodel
    for lrmap in 0.1 0.01
    do
        echo "mode for learning rate : " $lrmap
        python  -u modeinit.py --nmodel $nmodel --lr_map $lrmap --nclimb $nclimb &
    done
    wait
done
