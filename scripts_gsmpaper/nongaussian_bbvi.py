import numpy as np
import sys, os, time
import argparse
import matplotlib.pyplot as plt
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#
sys.path.append('../src/')
import bbvi_pdbmodels as bbvi
import gaussianq
import divergences as divs
import diagnostics as dg

import pdbmodels
import folder_path
import modeinit
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('--skewness', type=float, default=0.0, help='rank')
parser.add_argument('--tailw', type=float, default=1.0, help='rank')
#arguments for BBVI
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=8, help='batch size, default=8')
parser.add_argument('--lr', type=float, default=0.01, help='lr for bbvi')
parser.add_argument('--mode', type=str, default="full", help='which mode of the alg to use')
#Arguments for qdist
parser.add_argument('--scale', type=float, default=1., help='scale of Gaussian to initilize')
parser.add_argument('--covinit', type=str, default='identity', help='initialize covariance at identity')
#Arguments for hill climb
parser.add_argument('--modeinit', type=int, default=0, help='start from mode')
parser.add_argument('--lr_map', type=float, default=0.01, help='lr for hill climb')
parser.add_argument('--nclimb', type=int, default=1000, help='number of iterations for hill climb')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

print()
args = parser.parse_args()
D = args.D

np.random.seed(100)
loc = np.random.random(D).astype(np.float32)
scale = np.random.uniform(0.5, 2., D).astype(np.float32)
skewness = args.skewness
tailweight = args.tailw
basemodel = gaussianq.FR_Gaussian_cov(d=D, mu=loc, scale=scale, dtype=tf.float32)
model = gaussianq.SinhArcsinhTransformation(d=D, loc=0., scale=1.,
                                            skewness=skewness, tailweight=tailweight,
                                            distribution=basemodel.q,
                                            dtype=tf.float32)
samples = tf.constant(model.sample(1000))
print('samples shape : ', samples.shape)
idx = list(np.arange(D)[:int(min(D, 10))])

###
modelname = f'd{D}-s{skewness:0.1f}-t{tailweight:0.1f}'
modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/ArchSinh/'
folderpath = folder_path.bbvi_path(args)
savepath = f"{modelpath}/{folderpath}/{modelname}/S{args.seed}/"
os.makedirs(savepath, exist_ok=True)
print("\nsave in path : %s"%savepath)
np.save(savepath + 'mu', model.loc.numpy())
np.save(savepath + 'scale', model.scale.numpy())
suptitle = modelname + "-".join(savepath.split('/')[-3:-1])
print('\nSuptitle : ', suptitle)

######################
### MAP esimtate for initialization
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)
if args.modeinit:
    x0 = modeinit.mode_init(x0, model, args, modelpath)        
print("initialize at : ", x0)

###########
def callback(qdist, ls, i):
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000).numpy()
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)


def train(qdist, model, lr=1e-3, mode='full', batch_size=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None):

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    val_and_grad_func = bbvi.parse_mode(mode)
    elbos, losses, qdivs, counts, fdivs = [], [], [], [], []
    gradcount = 0
    for epoch in range(niter+1):

        elbo, loss, grads = val_and_grad_func(qdist, model, batch=tf.constant(batch_size))
        elbo = elbo.numpy()
        if np.isnan(elbo):
            print("NaNs!!! :: ", epoch, elbo)
            break

        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        elbos.append(elbo)
        losses.append(loss)
        gradcount += batch_size
        counts.append(gradcount) 
        qdiv = divs.qdivergence(qdist, model, nsamples=1000)
        qdivs.append(qdiv)
        if samples is not None:
            fdiv = divs.fkl_divergence(qdist, model, samples=samples)
            fdivs.append(fdiv)
        else:
            fdivs = None
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            if callback is not None: callback(qdist, [np.array(elbos)], epoch)

    print("return")
    return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)
    


### Setup VI
print()
print("Start VI")
qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=args.scale)
if args.covinit == 'noise':
    raise NotImplementedError
print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))

qdist, losses, elbos, qdivs, counts, fdivs = train(qdist, model,
                                                        mode=args.mode,
                                                        batch_size=args.batch, 
                                                        lr=args.lr, 
                                                        niter=args.niter,
                                                        nprint=args.niter//10,
                                                        callback=callback,
                                                        samples=samples)
#
print(losses.shape, elbos.shape, qdivs.shape, counts.shape)
print("number of gradient calls in BBVI : ", counts[-1])
dg.plot_bbvi_losses(elbos, qdivs, savepath, suptitle=suptitle)
np.save(savepath + 'elbo', elbos)
np.save(savepath + 'qdivs', qdivs)
np.save(savepath + 'losses', losses)
np.save(savepath + 'counts', counts)
np.save(savepath + 'fdivs', fdivs)
    

qsamples = qdist.sample(1000)
qsamples = qsamples.numpy()
dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist', suptitle=suptitle)
np.save(savepath + 'mufit', qdist.loc.numpy())
np.save(savepath + 'scalefit', qdist.scale.numpy())


