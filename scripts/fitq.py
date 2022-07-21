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
import bbvi, polyakvi, gaussianq
import diagnostics as dg

import models
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modelname', type=str, help='which model for the problem')
parser.add_argument('--alg', type=str, help='which algorithm to use')
parser.add_argument('--mode', type=str, help='which mode of the alg to use')
#arguments for flow
parser.add_argument('--nlayers', type=int, default=5, help='number of layers, default=5')
parser.add_argument('--nhidden', type=int, default=32, help='number of hddden params, default=32')
parser.add_argument('--qmodel', type=str, default="maf", help='model, one of maf, nsf or mdn, default=maf')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--niter', type=int, default=5001, help='number of iterations in training')
parser.add_argument('--nsamples', type=int, default=32, help='batch size, default=128')
#arguments for BBVI
parser.add_argument('--lr', type=float, default=0.001, help='lr for bbvi')
#arguments for PolyakVI
parser.add_argument('--epsmax', type=float, default=0.001, help='maximum step size for polyakvi')
parser.add_argument('--beta', type=float, default=0., help='momentum for polyakvi')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma factor for polyakvi')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

args = parser.parse_args()

savepath = './tmp/'
savepath = '//mnt/ceph/users/cmodi/polyakVI/%s/'%args.modelname
savepath += '%s-%s-%s/'%(args.alg, args.mode, args.qmodel)
if args.alg == 'bbvi': savepath += 'lr%0.e/'%args.lr
elif args.alg == 'polyak': savepath += 'g%0.1f-b%.1f/'%(args.gamma, args.beta)

os.makedirs(savepath, exist_ok=True)
print("\nsave in path %s"%savepath)


######################
print("Which model to use?")
model = getattr(models, args.modelname)()
print(model)
#model = models.PDB_1()
D = model.D
dg.corner(model.samples)

if args.qmodel == 'maf': qdist = flows.MAFFlow(D, nlayers=args.nlayers)
elif args.qmodel == 'mfg': qdist = gaussianq.MF_Gaussian(D, scale=1)
elif args.qmodel == 'frg': qdist = gaussianq.FR_Gaussian(D, scale=1)
else: 
  print('Variational model has to be one of maf, mfg or frg')

x0 = np.random.random(D).reshape(1, D)
print(qdist(x0))
if args.alg == 'polyak': suptitle = "-".join([args.alg, args.mode, args.qmodel, "g%0.1f"%args.gamma, "b%0.1f"%args.beta])
else: suptitle = "-".join([args.alg, args.mode, args.qmodel, "lr%0.e"%args.lr])
print('\nSuptitle : ', suptitle)


def callback(qdist, ls, i):
  if args.alg == 'polyak':
    dg.plot_polyak_losses(*ls, savepath, savename='losses_%04d'%i, suptitle=suptitle)
  else:
    dg.plot_bbvi_losses(*ls, savepath, savename='losses_%04d'%i, suptitle=suptitle)
  qsamples = model.gensamples(qdist)
  dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)

  


if args.alg == 'polyak':
  qdist, losses, elbos, epss = polyakvi.train(qdist, model.loglik, 
                                              mode=args.mode, 
                                              nsamples=args.nsamples, 
                                              epsmax=args.epsmax, 
                                              niter=args.niter, 
                                              gamma=args.gamma, 
                                              beta=args.beta, 
                                              callback=callback)

  dg.plot_polyak_losses(losses, elbos, epss, savepath, suptitle=suptitle)
  np.save(savepath + 'losses', losses)
  np.save(savepath + 'elbo', elbos)
  np.save(savepath + 'eps', epss)

elif args.alg == 'bbvi':
  qdist, elbos = bbvi.train(qdist, model.loglik, 
                            mode=args.mode,
                            nsamples=args.nsamples, 
                            lr=args.lr, 
                            niter=args.niter, 
                            callback=callback)
  #
  dg.plot_bbvi_losses(elbos, savepath, suptitle=suptitle)
  np.save(savepath + 'elbo', elbos)



#qdist.save_weights(savepath + 'model')
#tf.saved_model.save(qdist, savepath + 'model')
qsamples = model.gensamples(qdist)
np.save(savepath + 'samples', qsamples)
dg.compare_hist(qsamples, model.samples, savepath=savepath, savename='hist', suptitle=suptitle)


