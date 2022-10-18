import numpy as np
import sys, os
import argparse
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#
sys.path.append('../../normalizing_flows/src/')
import flows

sys.path.append('../src/')
import bbvi, polyakvi, mapp, hmc
import gaussianq
import diagnostics as dg

import pdbmodels
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmodel', type=int, help='which PDB model')
parser.add_argument('--alg', type=str, help='which algorithm to use')
parser.add_argument('--mode', type=str, help='which mode of the alg to use')
#arguments for flow
parser.add_argument('--nlayers', type=int, default=5, help='number of layers, default=5')
parser.add_argument('--nhidden', type=int, default=32, help='number of hddden params, default=32')
parser.add_argument('--qmodel', type=str, default="frg", help='model, one of maf, nsf or mdn, default=maf')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=5001, help='number of iterations in training')
parser.add_argument('--nsamples', type=int, default=32, help='batch size, default=32')
#arguments for BBVI
parser.add_argument('--lr', type=float, default=0.001, help='lr for bbvi')
#arguments for PolyakVI
parser.add_argument('--epsmax', type=float, default=0.001, help='maximum step size for polyakvi')
parser.add_argument('--beta', type=float, default=0., help='momentum for polyakvi')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma factor for polyakvi')
parser.add_argument('--llambda', type=float, default=1.0, help='lambda for slack polyakvi')
parser.add_argument('--delta', type=float, default=0.1, help='delta for slack polyakvi')
#parser.add_argument('--slack', type=int, default=0, help='slack polyakvi or regular polyakvi')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

args = parser.parse_args()
badmodels = [11, 33, 40, 51, 68, 69, 77]

###Check first if its a valid model
modelname = "PDB_%d"%args.nmodel
#model, refsamples = pdbmodels.setup_pdb_model(1)
model = pdbmodels.PDBModel_TF(args.nmodel)
print(model)
D = model.dims
dg.corner(model.samples, savepath='//mnt/ceph/users/cmodi/polyakVI/%s/'%modelname)
np.save('//mnt/ceph/users/cmodi/polyakVI/%s/samples'%modelname, model.samples) 

###
savepath = '//mnt/ceph/users/cmodi/polyakVI/%s/'%modelname
savepath += '%s-%s-%s/'%(args.alg, args.mode, args.qmodel)
if args.alg == 'bbvi': savepath += 'lr%0.e/'%args.lr
elif args.alg == 'polyak': savepath += 'g%0.1f-b%.1f-em%.e/'%(args.gamma, args.beta, args.epsmax)
elif args.alg == 'solyak': savepath += 'l%0.e-d%.ef/'%(args.llambda, args.delta)
if args.suffix !="": savepath = savepath[:-1] + "-" + args.suffix  + '/'

os.makedirs(savepath, exist_ok=True)
print("\nsave in path %s"%savepath)
suptitle = "PDB%d-"%args.nmodel + "-".join(savepath.split('/')[-3:-1])
print('\nSuptitle : ', suptitle)


######################
np.random.seed(0)
print("Which model to use?")

### MAP esimtate for initialization
# x0 = np.random.random(D).reshape(1, D).astype(np.float32) 
# #x0[0, -1] = 1. #model.y.numpy().std()
# print("Start at : ", x0)
# x0 = tf.Variable(x0)
# x0, losses = mapp.train(x0, model.log_likelihood_and_grad, niter=10001, lr=1e-3)
# print("Raw Mapp value : ", x0)
# print("True Mapp value : ", model.constrain_samples(x0))
# print("Posterior mean : ", model.samples.mean(axis=0))

## HMC sample for initialization
x0 = model.samples[-1:].copy()
print("Sample from HMC : ", x0)
print("Posterior mean : ", model.samples.mean(axis=0))
if args.nmodel in badmodels:
    x0 = np.random.random(D).reshape(1, D).astype(np.float32)*0.
else:
    x0 = model.unconstrain_samples(x0).astype(np.float32)
print("Raw value : ", x0)
############
### Setup VI
print()

scale = 1.
if args.nmodel in badmodels: scale = 0.1
#if args.qmodel == 'maf': qdist = flows.MAFFlow(D, nlayers=args.nlayers, mu=tf.constant(x0[0]), fitmean=True)e
if args.qmodel == 'maf': qdist = flows.MAFFlow(D, nlayers=args.nlayers,  fitmean=False)
elif args.qmodel == 'aff': qdist = flows.AffineFlow(D)
elif args.qmodel == 'mfg': qdist = gaussianq.MF_Gaussian(D, mu=tf.constant(x0[0]), scale=scale)
elif args.qmodel == 'frg': qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=scale)
else: 
    print('Variational model has to be one of maf, mfg or frg')
print('log prob : ', qdist.log_prob(np.random.random(D).reshape(1, D).astype(np.float32)))



# warmup = True
# while warmup:
#     print("\nWarmup")
#     try:
#         _, _ = bbvi.train(qdist, model.log_likelihood_and_grad, mode=args.mode, lr=args.lr, niter=100) 
#         warmup = False
#     except:
#         if args.qmodel == 'mfg': 
#             qdist = gaussianq.MF_Gaussian(D, mu=tf.constant(x0[0]), scale=qdist.scale/2.)
#         elif args.qmodel == 'frg': 
#             qdist = gaussianq.FR_Gaussian(D, mu=tf.constant(x0[0]), scale=qdist.scale/2.)
#         else:
#             raise NotImplementedError 
#         print("Retry with qdist scale : ", qdist.scale)


###########
def callback(qdist, ls, i):
    if args.alg == 'polyak' or args.alg == 'solyak':
        dg.plot_polyak_losses(*ls, savepath, savename='losses_%04d'%i, suptitle=suptitle)
    else:
        dg.plot_bbvi_losses(*ls, savepath, savename='losses_%04d'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000)
    qsamples = model.constrain_samples(qsamples).numpy()
    dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)


print("Start VI")
if args.alg == 'polyak' or args.alg == 'solyak':
  slack = 0
  if args.alg == 'solyak': slack = 1
  qdist, losses, elbos, epss = polyakvi.train(qdist, model.log_likelihood_and_grad,
                                              mode=args.mode, 
                                              nsamples=args.nsamples, 
                                              niter=args.niter, 
                                              callback=callback, 
                                              slack=slack, 
                                              epsmax=args.epsmax, 
                                              gamma=args.gamma, 
                                              beta=args.beta, 
                                              delta=args.delta,
                                              llambda=args.llambda
                                            )

  dg.plot_polyak_losses(losses, elbos, epss, savepath, suptitle=suptitle)
  np.save(savepath + 'losses', losses)
  np.save(savepath + 'elbo', elbos)
  np.save(savepath + 'eps', epss)

elif args.alg == 'bbvi':
    qdist, losses, elbos = bbvi.train(qdist, model.log_likelihood_and_grad,
                              mode=args.mode,
                              nsamples=args.nsamples, 
                              lr=args.lr, 
                              niter=args.niter, 
                              callback=callback)
    #
    dg.plot_bbvi_losses(elbos, losses, savepath, suptitle=suptitle)
    np.save(savepath + 'elbo', elbos)
    np.save(savepath + 'losses', losses)
    
elif args.alg == 'hmc':
    qsamples, accepted, probs, counts = hmc.sample(model.loglik, grad_log_prob=None, 
                                                   step_size=0.001, 
                                                   Nleapfrog=100, 
                                                   D=model.D, 
                                                   nsamples=1000, 
                                                   burnin=100,
                                                   q0=x0, nchains=4)
    #
    qdist = None
    print("Fraction accepted: ", accepted.sum()/accepted.size)
    print(qsamples.shape)
    qsamples = np.squeeze(qsamples).reshape(-1, model.D)
    qsamples = model.transform_samples(qsamples)
    

if qdist is not None:
    qsamples = qdist.sample(1000)
    qsamples = model.constrain_samples(qsamples).numpy()

np.save(savepath + 'samples', qsamples)
dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist', suptitle=suptitle)


