import os, sys

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done

##python -u fitq.py --modelname PDB_1 --alg polyak --mode score --beta 0.3 --niter 10001

time=20

slurmenv =  '\n'.join([
    '',
    'module --force purge',
    'module load modules-traditional',
    'module load cuda/11.0.3_450.51.06',
    'module load cudnn/v8.0.4-cuda-11.0',
    'module load slurm',
    'module load gcc',
    'module load openmpi',
    'source activate defpyn',
    ''])
    

def polyakvi_job(modelname, mode, beta, niter, qmodel, nsamples):
    
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    logname = 'polyak-%s'%(mode)
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p gpu',
        '#SBATCH --gpus=v100-32gb:1'
        '#SBATCH -c 1',
        '#SBATCH -t %d'%time,
        '#SBATCH -J polyvi',
        '#SBATCH -o logs/%s'%(logname+".o%j"),
        ''])
    slurm = slurm + slurmenv
    slurm = slurm + '\n'.join([
        '',
        'modelname=%s'%modelname,
        'alg=polyak',
        'mode=%s'%mode,
        'beta=%0.1f'%beta,
        'niter=%d'%niter,
        'qmodel=%s'%qmodel,
        'nsamples=%d'%nsamples,
        '',
        'time srun python -u fitq.py --modelname $modelname --alg $alg --mode $mode --niter $niter --beta $beta --qmodel $qmodel --nsamples $nsamples',
        ''
    ])
    
    f = open('polyakvi.slurm', 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch polyakvi.slurm')
    #os.system('rm polyakvi.slurm')
    return None



def bbvi_job(modelname, mode, lr, niter, qmodel, nsamples):
    
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    logname = 'bbvi-%s'%(mode)
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p gpu',
        '#SBATCH --gpus=v100-32gb:1'
        '#SBATCH -c 1',
        '#SBATCH -t %d'%time,
        '#SBATCH -J bbvi',
        '#SBATCH -o logs/%s'%(logname+".o%j"),
        ''])
    slurm = slurm + slurmenv
    slurm = slurm + '\n'.join([
        '',
        'modelname=%s'%modelname,
        'alg=bbvi',
        'mode=%s'%mode,
        'lr=%0.5f'%lr,
        'niter=%d'%niter,
        'qmodel=%s'%qmodel,
        'nsamples=%d'%nsamples,
        '',
        'time srun python -u fitq.py --modelname $modelname --alg $alg --mode $mode --niter $niter --lr $lr --qmodel $qmodel --nsamples $nsamples',
        ''
    ])
    
    f = open('bbvi.slurm', 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch bbvi.slurm')
    #os.system('rm bbvi.slurm')
    return None



modelname = 'PDB_1'
niter = 50001
nsamples = 128
qmodel = 'mfg'
##polyakvi_job(modelname, mode='score', beta=0., niter=niter, qmodel=qmodel)

for qmodel in ['mfg', 'frg', 'maf']:
    
    alg = 'polyak'
    for mode in ['full', 'score', 'scorenorm']:
         for beta in [0., 0.3]:
             print(modelname, alg, mode, beta, niter)
             #polyakvi_job(modelname, mode=mode, beta=beta, niter=niter, qmodel=qmodel, nsamples=nsamples)
             if qmodel == 'mfg': continue
             else: os.system('time python -u fitq.py --modelname %s --alg polyak --mode %s --niter %d --beta %0.1f --qmodel %s --nsamples %d'%(modelname, mode, niter, beta, qmodel, nsamples))


    alg = 'bbvi'
    for mode in ['full', 'path', 'score']:
    #for mode in [ 'path']:
         for lr in [1e-2, 1e-3, 1e-4]:
            print(modelname, alg, mode, lr, niter)
            #bbvi_job(modelname, mode=mode, lr=lr, niter=niter, qmodel=qmodel, nsamples=nsamples)
            os.system('time python -u fitq.py --modelname %s --alg bbvi --mode %s --niter %d --lr %0.5f --qmodel %s --nsamples %d'%(modelname, mode, niter, lr, qmodel, nsamples))
