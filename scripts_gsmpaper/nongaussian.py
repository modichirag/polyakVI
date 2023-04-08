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
import gsm
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
#arguments for GSM
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--qsample', type=int, default=1, help='sample from q distribution, default=1')
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
modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/ArchSinh/gsm-frg/%s/'%modelname
folderpath = folder_path.frg_path_gsm(args)
savepath = f"{modelpath}/{folderpath}/S{args.seed}/"
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
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d.png'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000).numpy()
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)


# @tf.function
# def qdivergence(qdist, model, nsamples=1000):
#     print('creating graph q divergence')
#     samples = qdist.sample(nsamples)
#     logp1 = qdist.log_prob(samples)
#     logp2 = model.log_prob(samples)
#     div = tf.reduce_mean(logp1 - logp2)
#     return div

def train(qdist, model, niter=1000, batch_size=4, nprint=10, dtype=tf.float32, callback=None, verbose=True, samples=None, qsample=True):

    qdivs, fdivs, counts = [], [], []
    gradcount = 0
    for epoch in range(niter + 1):

        if qsample:
            x = list(qdist.sample(batch_size))
        else:
            x = list(model.sample(batch_size))
        _ = gsm.gaussian_update_batch(x, model, qdist)
        if np.linalg.eigvals(qdist.cov).min() < 0:
            print("ERROR : NEGATIVE EIGENVALUES IN COVARIANCE")
            break
        gradcount += batch_size
        counts.append(gradcount)
        qdiv = divs.qdivergence(qdist, model, nsamples=1000)        
        qdivs.append(qdiv)
        fdiv = divs.fkl_divergence(qdist, model, samples)
        fdivs.append(fdiv)
        
        if (epoch %(niter//nprint) == 0) & verbose: 
            print("Loss at epoch %d is"%epoch, qdiv[0])
            if callback is not None: 
                callback(qdist, [np.array(qdivs)], epoch)

    return qdist, np.array(qdivs), np.array(counts), np.array(fdivs)


### Setup VI
print()
print("Start VI")
qdist = gaussianq.FR_Gaussian_cov(D, mu=tf.constant(x0[0]), scale=args.scale)
if args.covinit == 'noise':
    covinit = np.eye(D)
    for i in range(D):
        covinit[i, i] = np.abs(np.random.normal(0, args.scale))   
    qdist.cov.assign(covinit)

print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))


qdist, qdivs, counts, fdivs = train(qdist, model,
                                    batch_size=args.batch, 
                                    niter=args.niter, 
                                    callback=callback,
                                    samples=samples,
                                    qsample=args.qsample)

print("number of gradient calls in GSM : ", counts[-1])
print(qdivs.shape, fdivs.shape)
dg.plot_bbvi_losses(qdivs[..., 0], qdivs[..., 0], savepath, suptitle=suptitle)
np.save(savepath + 'qdivs', qdivs)
np.save(savepath + 'fdivs', fdivs)
np.save(savepath + 'counts', counts)


qsamples = qdist.sample(1000)
qsamples = qsamples.numpy()
np.save(savepath + 'mufit', qdist.loc.numpy())
np.save(savepath + 'covfit', qdist.cov.numpy())
dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist', suptitle=suptitle)


