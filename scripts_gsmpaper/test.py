import numpy as np
import os, sys
#
sys.path.append('../src/')
import pdbmodels
import diagnostics as dg
#

for nmodel in range(103):

    print()
    if nmodel in [20, 25]: continue
    modelname = "PDB_%d"%nmodel
    print(modelname)

    try:
        model = pdbmodels.PDBModel_TF(nmodel)
        D = model.dims

        #if D < 11: continue
        print("Dimensions of the problem : ", D)
        print("Model sample shape in constrained space : ", model.samples.shape[-1])

        if os.path.isfile(f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{nmodel}//gsm-frg/B02/S0/samples.npy'):
            f = np.load(f'/mnt/ceph/users/cmodi/polyakVI/gsmpaper/PDB_{nmodel}//gsm-frg/B02/S0/samples.npy')
            print("Model sample shape as sampled : ", f.shape[-1])

        idx = np.random.permutation(model.samples.shape[0])[:10000] #1000
        samples_fvi = model.samples[idx]
        unc_samples_fvi = model.unconstrain_samples(samples_fvi.astype(np.float32)).astype(np.float32)
        nqsamples = unc_samples_fvi.shape[0]
        print("shape of unconstrained samples: ", unc_samples_fvi.shape[-1])

        Dplot = min(D, 10)
        modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/%s/'%modelname
        dg.corner(model.samples[:, :Dplot], savepath=modelpath)
        dg.corner(unc_samples_fvi[:, :Dplot], savepath=modelpath, savename='corner-unc')
        
            
    except Exception as e:
        print("##EXCEPTION##\n", e)
