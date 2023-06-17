import numpy as np
import sys, os
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.append('../src/')
sys.path.append('../scripts/')
import modevi
import diagnostics as dg
import argparse
import pdbmodels

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nmodel', type=int, help='which PDB model')
#arguments for GSM
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--lr_map', type=float, default=0.01, help='lr for hill climb')
parser.add_argument('--nclimb', type=int, default=1000, help='number of iterations for hill climb')

def mode_init(x0, model, args, modelpath, verbose=False, redo=False):
    
    modefile = f"{modelpath}/modeinit/N{args.nclimb}-lr{args.lr_map:0.1e}"
    countfile = f"{modelpath}/modeinit/N{args.nclimb}-lr{args.lr_map:0.1e}-ngrads"
    try:
        mode = tf.constant(np.load(f"{modefile}.npy"))
        print(f"Loading mode from file : {modefile}.npy")
        if redo:
            raise 
    except Exception as e:
        print("Exception in loading mode", e)
        
        mode, loss = modevi.check_hill_climb(x0, model.log_likelihood_and_grad,
                                             lr=args.lr_map,
                                             niter=args.nclimb,
                                             early_stop=False,
                                             verbose=verbose)
        os.makedirs(f"{modelpath}/modeinit", exist_ok=True)
        np.save(f"{modefile}", mode)
        np.save(f"{countfile}", model.grad_count)
        dg.plot_mode(model.constrain_samples(mode), model.samples,
                     savepath=f"{modelpath}/modeinit/",
                     savename=f"N{args.nclimb}-lr{args.lr_map:0.1e}",
                     suptitle=f"PDB_{args.nmodel}-N{args.nclimb}-lr{args.lr_map:0.1e}")
        # plot loss
        plt.figure()
        plt.plot(loss, 'C0')
        plt.plot(-loss, 'C0--')
        plt.loglog()
        plt.grid(which='both')
        plt.suptitle(f"PDB_{args.nmodel}-N{args.nclimb}-lr{args.lr_map:0.1e}")
        plt.savefig(f"{modelpath}/modeinit/loss-N{args.nclimb}-lr{args.lr_map:0.1e}.png")
        plt.close()
        
    print("mode found at : ", mode)
    print("constrained mode : ", model.constrain_samples(mode))
    print("sample mean : ", model.samples.mean(axis=0))
    
    return mode



if __name__ == "__main__":


    print()
    args = parser.parse_args()
    modelname = "PDB_%d"%args.nmodel
    modelpath = '//mnt/ceph/users/cmodi/polyakVI/gsmpaper/%s/'%modelname
    model = pdbmodels.PDBModel_TF(args.nmodel)
    print(model)
    D = model.dims
    print("Dimensions of the problem : ", D)
    print("Model sample shape in constrained space : ", model.samples.shape)

    ### MAP esimtate for initialization
    np.random.seed(args.seed)
    x0 = np.random.random(D).reshape(1, D).astype(np.float32)
    x0 = tf.constant(x0)
    x0 = mode_init(x0, model, args, modelpath, verbose=True, redo=True)
    
