import numpy as np
import sys, os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('../src/')
import diagnostics as dg

import pdbmodels
#


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmodel', type=int, help='which PDB model')

args = parser.parse_args()
n = args.nmodel
try:
     modelname = pdbmodels.pos[n]
except:
     modelname = "PDB{n}"
print(modelname)

if os.path.isfile(f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B02/S0/samples.npy') or \
   os.path.isfile(f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}///bbvi-frg-full/B02-lr1.0e-02/S0/samples.npy') :
     D = np.load(f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B02/S0/samples.npy').shape[1]
     print(f"Dimensions : {D}")
else:
     print("No samples found for fiducial runs, B=2")
     sys.exit()
     

savepath = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//figures/'
os.makedirs(savepath, exist_ok=True)

def loadloss(path, mode, maxseed=1):
     lls, counts = [], []
     for i in range(maxseed):
          try:
               losses = np.load(f'{path}/S{i}/{mode}.npy')[...,0]
               counts = np.load(f'{path}/S{i}/counts.npy')
               lls.append(losses)
          except Exception as e:
               print(e)
     lls = np.array(lls)
     return lls, counts

     
def lossplot(n, mode):

     def _plot_loss(axis, path, color, ls, maxseed=11, lbl=None):
         lls, counts = [], []
         for i in range(maxseed):
              try:
                   losses = np.load(f'{path}/S{i}/{mode}.npy')[...,0]
                   counts = np.load(f'{path}/S{i}/counts.npy')
                   axis.plot(counts, losses, color=color, ls=ls, alpha=0.2)
                   lls.append(losses)
              except Exception as e:
                   print(e)
         lls = np.array(lls)
         try:
              axis.plot(counts, lls.mean(axis=0), color=color, ls=ls, lw=2, label=lbl)
         except:
              pass
     fig, ax = plt.subplots(1, 2, figsize=(9, 4))

     #for gsm
     print()
     print("GSM")
     for ib, b in enumerate([2, 4]):
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B{b:02d}/'
          _plot_loss(ax[0], path, f'C{ib}', ls='-')
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B{b:02d}-modeinit/'
          _plot_loss(ax[1], path, f'C{ib}', ls='-')
               
               
     #for bbvi
     print()
     print("BBVI")
     for ib, b in enumerate([ 2, 4, 8]):
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//bbvi-frg-full/B{b:02d}-lr1.0e-02/'
          _plot_loss(ax[0], path, f'C{ib}', ls='--')
          bbvipath = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//bbvi-frg-full/B{b:02d}-lr1.0e-02-modeinit/'
          _plot_loss(ax[1], path, f'C{ib}', ls='--')
               

     for axis in ax:
          axis.semilogx()
          axis.grid(which='both', lw=0.3)
          axis.set_xlabel('# Gradient evaluations')
     if mode == 'qdivs':
          ax[0].set_ylabel('$\sum_{x\sim q}(\log\, q - \log\, p)$')
     if mode == 'fdivs':
          ax[0].set_ylabel('$\sum_{x\sim p}(\log\, q - \log\, p)$')
     axis.legend(bbox_to_anchor=(1, 1))
     plt.suptitle(f'Model{n}: {modelname}\nD={D}', fontsize=10)
     plt.tight_layout()
     plt.savefig(f'{savepath}/{mode}.png')     
     for axis in ax:
          try: axis.loglog()
          except: pass
     plt.savefig(f'{savepath}/{mode}-log.png')

     
def compare_hist(n):     

     model = pdbmodels.PDBModel_TF(n)
     ref_samples = model.samples
     D = min(model.dims, ref_samples.shape[1])
     Dtrue = D*1
     D = min(D, 10)
     
     nbins = 20 
     fig, ax = plt.subplots(2, D, figsize=(3*D, 3))
     for ii in range(D):
          ax[0, ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, lw=2, histtype='step', color='k', label='Ref');
          ax[1, ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, lw=2, histtype='step', color='k', label='Ref');
          
     def _plot_hist(axis, path, nbins, alpha, color, histtype='stepfilled', label=None):
          for i in range(11): 
               try:
                    samples = np.load(f'{path}/S{i}/samples.npy')
                    for ii in range(D):
                         axis[ii].hist(samples[:, ii], bins=nbins, density=True, alpha=alpha, label=label, color=color, histtype=histtype);
               except Exception as e:
                    print(e)
               
     lbl = None
     for ib, b in enumerate([2]):
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B{b:02d}/'
          _plot_hist(ax[0], path, nbins=nbins, alpha=0.1, label=lbl, color=f'C{ib}')
                     
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//gsm-frg/B{b:02d}-modeinit/'
          _plot_hist(ax[0], path, nbins=nbins, alpha=0.2, label=lbl, color=f'C{ib}', histtype='step')
               
          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//bbvi-frg-full/B{b:02d}-lr1.0e-02/'
          _plot_hist(ax[1], path, nbins=nbins, alpha=0.1, label=lbl, color=f'C{ib}')

          path = f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{n}//bbvi-frg-full/B{b:02d}-lr1.0e-02-modeinit/'
          _plot_hist(ax[1], path, nbins=nbins, alpha=0.5, label=lbl, color=f'C{ib}', histtype='step')

     plt.suptitle(f'Model{n}: {modelname}\nD={Dtrue}', fontsize=10)
     plt.tight_layout()
     plt.savefig(f'{savepath}/hist.png')
     plt.close()

lossplot(args.nmodel, 'qdivs')
#lossplot(args.nmodel, 'fdivs')
#compare_hist(args.nmodel)
