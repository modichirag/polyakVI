import numpy as np
import sys, os
import tensorflow as tf

sys.path.append('../src/')
import modevi
import diagnostics as dg


def mode_init(x0, model, args, modelpath):
    
    modefile = f"{modelpath}/modeinit/N{args.nclimb}-lr{args.lr_map:0.1e}"
    countfile = f"{modelpath}/modeinit/N{args.nclimb}-lr{args.lr_map:0.1e}-ngrads"
    try:
        mode = tf.constant(np.load(f"{modefile}.npy"))
        print(f"Loading mode from file : {modefile}.npy")
        
    except Exception as e:
        print("Exception in loading mode", e)
        
        mode, _ = modevi.check_hill_climb(x0, model.log_likelihood_and_grad,
                                          lr=args.lr_map,
                                          niter=args.nclimb,
                                          early_stop=False,
                                          verbose=False)
        os.makedirs(f"{modelpath}/modeinit", exist_ok=True)
        np.save(f"{modefile}", mode)
        np.save(f"{countfile}", model.grad_count)
        dg.plot_mode(model.constrain_samples(mode), model.samples,
                     savepath=f"{modelpath}/modeinit/",
                     savename=f"N{args.nclimb}-lr{args.lr_map:0.1e}")
        
    print("mode found at : ", mode)
    print("constrained mode : ", model.constrain_samples(mode))
    print("sample mean : ", model.samples.mean(axis=0))
    
    return mode
