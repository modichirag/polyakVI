print()
import numpy as np
import sys, os
import argparse
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#
sys.path.append('../src/')
import bbvi_pdbmodels as bbvi
import gaussianq
import diagnostics as dg
import divergences as divs

import pdbmodels
import folder_path
import modeinit
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmodel', type=int, help='which PDB model')
parser.add_argument('--mode', type=str, default="full", help='which mode of the alg to use')
#arguments for BBVI
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=0.01, help='lr for bbvi')
#Arguments for qdist
parser.add_argument('--scale', type=float, default=1., help='scale of Gaussian to initilize')
parser.add_argument('--covinit', type=str, default='identity', help='initialize covariance at identity')
#Hill climb init
parser.add_argument('--modeinit', type=int, default=1, help='start from mode')
parser.add_argument('--lr_map', type=float, default=0.01, help='lr for hill climb')
parser.add_argument('--nclimb', type=int, default=1000, help='number of iterations for hill climb')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--log', type=int, default=1, help='log print statements to file, default=0')
parser.add_argument('--err', type=int, default=1, help='log err to file, default=0')

print()
args = parser.parse_args()
modelname = "PDB_%d"%args.nmodel
model = pdbmodels.PDBModel_TF(args.nmodel)
print(model)
D = model.dims
Dplot = min(10, D)

np.random.seed(99)
idx = np.random.permutation(model.samples.shape[0])[:1000]
try:
    samples_fvi = model.samples[idx]
    unc_samples_fvi = tf.constant(model.unconstrain_samples(samples_fvi).astype(np.float32))
    unc_samples = model.unconstrain_samples(model.samples).astype(np.float32)
except Exception as e:
    print(e)
    unc_samples_fvi = None
###
modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/%s/'%modelname
os.makedirs(modelpath, exist_ok=True)
dg.corner(model.samples[:, :Dplot], savepath=modelpath)
np.save(modelpath + 'samples', model.samples) 

folderpath = folder_path.bbvi_path(args)
savepath = f"{modelpath}/{folderpath}/S{args.seed}/"
os.makedirs(savepath, exist_ok=True)
print("\nsave in path %s"%savepath)
suptitle = "PDB%d-"%args.nmodel + "-".join(savepath.split('/')[-3:-1])
print('\nSuptitle : ', suptitle)
if args.err: sys.stderr = open(f'{savepath}/run.err', 'w')
if args.log: sys.stdout = open(f'{savepath}/run.log', 'w')


######################
###########
def callback(qdist, ls, i):
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000)
    qsamples = model.constrain_samples(qsamples).numpy()
    dg.compare_hist(qsamples[:, :Dplot], model.samples[:, :Dplot], savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)


### MAP esimtate for initialization
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)
if args.modeinit == 1:
    x0 = modeinit.mode_init(x0, model, args, modelpath)
elif args.modeinit == 2:
    #x0 = model.uncomstrain_samples(model.samples[-1])
    x0 = tf.constant(model.unconstrain_samples(model.samples.mean(axis=0)), dtype=tf.float32)
    
print("number of gradient calls in setup : ", model.grad_count)
model.reset_gradcount()
print("initialize at : ", x0)
        

############
### Setup VI
print()
print("Start VI")

def fit(x0, batch_size, scale):
    qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=scale)
    if args.covinit == 'noise':
        raise NotImplementedError
    print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))

    qdist, losses, elbos, qdivs, counts, fdivs = bbvi.train_frg(qdist, model,
                                                            mode=args.mode,
                                                            batch=batch_size, 
                                                            lr=args.lr, 
                                                            niter=args.niter,
                                                            nprint=args.niter//10,
                                                            callback=callback,
                                                            samples=unc_samples_fvi,
                                                            savepath=savepath)
    return qdist, losses, elbos, qdivs, counts, fdivs


#
run = 1
for batch in [args.batch, int(args.batch*2), int(args.batch*4)]:
    for scale in [args.scale, args.scale/10., args.scale/100.]:
        try:
            if run == 1:
                print(f"Try for batch factor = {batch/args.batch}, scale factor = {scale/args.scale}")
                model.reset_gradcount()
                qdist, losses, elbos, qdivs, counts, fdivs = fit(x0, batch, scale)
                run = 0
        except Exception as e:
            print(e)
    
#
print(losses.shape, elbos.shape, qdivs.shape, counts.shape)
print("number of gradient calls in BBVI : ", model.grad_count)
dg.plot_bbvi_losses(elbos, qdivs, savepath, suptitle=suptitle)
np.save(savepath + 'elbo', elbos)
np.save(savepath + 'qdivs', qdivs)
np.save(savepath + 'losses', losses)
np.save(savepath + 'counts', counts)
np.save(savepath + 'fdivs', fdivs)
    

qsamples = qdist.sample(1000)
qsamples = model.constrain_samples(qsamples).numpy()
dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist', suptitle=suptitle)
np.save(savepath + 'samples', qsamples)
np.save(savepath + 'mufit', qdist.loc.numpy())
np.save(savepath + 'scalefit', qdist.scale.numpy())


