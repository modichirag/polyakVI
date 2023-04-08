import numpy as np
import sys, os
import argparse

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


sys.path.append('../src/')
import bbvi_pdbmodels as bbvi
import gaussianq
import diagnostics as dg

import pdbmodels
import folder_path
import modeinit
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmodel', type=int, help='which PDB model')
#arguments for FVI
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=0.01, help='lr for fvi')
#Arguments for qdist
parser.add_argument('--scale', type=float, default=1., help='scale of Gaussian to initilize')
parser.add_argument('--covinit', type=str, default='identity', help='initialize covariance at identity')
#Hill climb init
parser.add_argument('--modeinit', type=int, default=1, help='start from mode')
parser.add_argument('--lr_map', type=float, default=0.01, help='lr for hill climb')
parser.add_argument('--nclimb', type=int, default=1000, help='number of iterations for hill climb')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

print()
args = parser.parse_args()
modelname = "PDB_%d"%args.nmodel
model = pdbmodels.PDBModel_TF(args.nmodel)
print(model)
D = model.dims
idx = np.random.permutation(model.samples.shape[0])[:1000]
samples_fvi = model.samples[idx]
unc_samples_fvi = model.unconstrain_samples(samples_fvi).astype(np.float32)
unc_samples = model.unconstrain_samples(model.samples).astype(np.float32)

###
modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/%s/'%modelname
dg.corner(model.samples, savepath=modelpath)
np.save(modelpath + 'samples', model.samples) 

folderpath = folder_path.fvi_path(args)
savepath = f"{modelpath}/{folderpath}/"
os.makedirs(savepath, exist_ok=True)
print("\nsave in path %s"%savepath)
suptitle = "PDB%d-"%args.nmodel + "-".join(savepath.split('/')[-3:-1])
print('\nSuptitle : ', suptitle)


######################
### MAP esimtate for initialization
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)
if args.modeinit == 1:
    x0 = modeinit.mode_init(x0, model, args, modelpath)
elif args.modeinit == 2:
    x0 = tf.constant(model.unconstrain_samples(model.samples.mean(axis=0)), dtype=tf.float32)
    
print("number of gradient calls in setup : ", model.grad_count)
model.reset_gradcount()
print("initialize at : ", x0)
        
###########
def callback(qdist, ls, i):
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000)
    qsamples = model.constrain_samples(qsamples).numpy()
    dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)


# #######
@tf.function
def fvi(samples, qmodel):
    print("create graph fvi")
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(qmodel.trainable_variables)
        logq = qmodel.log_prob(samples)
        f = -tf.reduce_mean(logq)

    gradients = tape.gradient(f, qmodel.trainable_variables)
    return f, gradients

@tf.function
def update(opt, grads, qmodel):
    opt.apply_gradients(zip(grads, qmodel.trainable_variables))
    

######
def train(qdist, model, lr=1e-3, batch=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None):

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    losses, qdivs, fdivs, counts = [], [], [], []
    
    for epoch in range(niter+1):

        idx = np.random.permutation(samples.shape[0])[:batch]
        batch_samples = tf.constant(unc_samples[idx])
        loss, grads = fvi(batch_samples, qdist)
        update(opt, grads, qdist)
        #opt.apply_gradients(zip(grads, qdist.trainable_variables))

        losses.append(loss)
        qdiv = bbvi.qdivergence(qdist, model)
        qdivs.append(qdiv)
        if samples is not None:
            fdiv = bbvi.fkl_divergence(samples, qdist)
            fdivs.append(fdiv)
        counts.append(model.grad_count)
        
        if (epoch %nprint == 0) & verbose: 
            print("Loss at epoch %d is"%epoch, loss)
            if callback is not None: callback(qdist, [np.array(losses)], epoch)
            
            
    return qdist, np.array(losses), np.array(fdivs), np.array(qdivs), np.array(counts)


############
### Setup VI
print()
print("Start VI")
#qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=args.scale)
qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]))
print("\nTrainable varaiables : ", qdist.trainable_variables)
if args.covinit == 'noise':
    raise NotImplementedError
print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))

qdist, losses, fdivs, qdivs, counts = train(qdist, model,
                                                   batch=args.batch, 
                                                   lr=args.lr, 
                                                   niter=args.niter,
                                                   nprint=args.niter//10,
                                                   callback=callback,
                                                   samples=tf.constant(unc_samples_fvi))
#
print("number of gradient calls in FVI : ", model.grad_count)
dg.plot_bbvi_losses(losses, qdivs, savepath, suptitle=suptitle)
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


