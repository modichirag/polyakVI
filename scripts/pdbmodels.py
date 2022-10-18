import numpy as np
import sys, os
import json
import tensorflow as tf
#import MCMC as mcmc

from posteriordb import PosteriorDatabase

##Setup your posterior DB environment
PDBPATH = os.path.join('/mnt/home/cmodi/Research/Projects/posteriordb/posterior_database/')
CMDSTAN = '/mnt/home/cmodi/Research/Projects/cmdstan/'
BRIDGESTAN = '/mnt/home/cmodi/Research/Projects/bridgestan2/'
MODELDIR = '../compiled_models/'

sys.path.append(BRIDGESTAN)
import PythonClient as pbs

def get_pdb_model(model_n):

    pdb = PosteriorDatabase(PDBPATH)
    modelnames = pdb.posterior_names()
    posterior = pdb.posterior(modelnames[model_n])
    posname = posterior.name
    print("Model name :", posname)

    stanmodel, data = posterior.model, posterior.data.values()

    refdrawsdict = posterior.reference_draws()
    keys = refdrawsdict[0].keys()
    stansamples = []
    for key in keys:
        stansamples.append(np.array([refdrawsdict[i][key] for i in range(len(refdrawsdict))]).flatten())
    samples = np.array(stansamples).copy().astype(np.float32).T

    return stanmodel, data, samples


def setup_pdb_model(model_n):

    stanmodel, data, samples = get_pdb_model(model_n)

    #Save stan model code
    modeldir = MODELDIR + '/PDB_%02d/'%model_n
    os.makedirs(modeldir, exist_ok=True)
    modpath = modeldir + 'PDB_%02d'%model_n
    modpath = os.path.abspath(modpath)
    print(modpath)
    with open(modpath + '.stan', 'w') as f:
        f.write(stanmodel.code('stan'))

    #Save corresponding data
    datapath = modpath + '.data.json'
    with open(datapath, 'w') as f:
        json.dump(data, f,)

    #Save compiled shared object
    sopath = modpath + '_model.so'

    try:
        if os.path.isfile(sopath):
            print("PDB_%02d_model.so file exists"%model_n)
        model = pbs.PyBridgeStan(sopath, datapath)
    except Exception as e:
        print(e)
        cwd = os.getcwd()
        print(cwd)
        os.chdir(BRIDGESTAN)
        os.system('make  %s'%sopath) 
        os.chdir(cwd)
        model = pbs.PyBridgeStan(sopath, datapath)
    
    return model, samples


class PDBModel_TF():
    
    def __init__(self, model_n):

        self.model_n = model_n
        self.model, self.samples = setup_pdb_model(model_n)
        self.dims = self.model.dims()


    # @tf.custom_gradient
    # def log_likelihood_and_grad(self, q):
        
    #     def np_loglik(x):
    #         lk, _ = self.model.log_density_gradient(np.float64(x))
    #         return np.float32(lk)

    #     def loglik(x):
    #         lk = tf.numpy_function(np_loglik, [tf.cast(x, tf.float64)], Tout=tf.float32)
    #         return lk

    #     def np_grad_loglik(x):
    #         _, lk_g = self.model.log_density_gradient(np.float64(x))
    #         return np.float32(lk_g)

    #     def grad_loglik(x):
    #         lk_g = tf.numpy_function(np_grad_loglik, [tf.cast(x, tf.float64)], Tout=tf.float32)
    #         return lk_g

    #     lk = []
    #     lk = tf.map_fn(loglik, q, fn_output_signature=tf.float32)
        
    #     def grad(upstream):
    #         lk_g = []
    #         lk_g = tf.map_fn(grad_loglik, q, fn_output_signature=tf.float32)
    #         return tf.expand_dims(upstream, -1) * lk_g
    #     return lk, grad



    @tf.custom_gradient
    def log_likelihood_and_grad(self, q):
        
        def np_loglik_and_grad(x):
            lk, lk_g = self.model.log_density_gradient(np.float64(x))
            return np.float32(lk), np.float32(lk_g)

        def loglik_and_grad(x):
            lk, lk_g = tf.numpy_function(np_loglik_and_grad, [tf.cast(x, tf.float64)], Tout=[tf.float32,tf.float32])
            return lk, lk_g

        lk, lk_g = tf.map_fn(loglik_and_grad, q, fn_output_signature=(tf.float32, tf.float32))
        
        def grad(upstream):
            return tf.expand_dims(upstream, -1) * lk_g
        return lk, grad


    def log_likelihood(self, q):
        
        if type(q) != np.ndarray:
            q = q.numpy().astype(np.float64)
            convert = True
        else: 
            q = q.astype(np.float64)
            convert = False

        if len(q.shape) > 1:
            lk = []
            for iq in q: 
                lk.append(self.model.log_density(iq))
            lk = np.array(lk).astype(np.float32)
        else:
            lk = self.model.log_density(q)

        if convert: lk = tf.constant(lk, dtype=tf.float32)
        return lk


    def grad_log_likelihood(self, q):
        
        if type(q) != np.ndarray:
            q = q.numpy().astype(np.float64)
            convert = True
        else: 
            q = q.astype(np.float64)
            convert = False

        if len(q.shape) > 1:
            lk = []
            for iq in q: 
                lk.append(self.model.log_density_gradient(iq)[1].astype(np.float32)*1.)
            lk = np.array(lk)
        else:
            lk = self.model.log_density_gradient(q)[1]

        if convert: lk = tf.constant(lk, dtype=tf.float32)
        return lk

    def constrain_samples(self, q):
        
        if type(q) != np.ndarray:
            q = q.numpy().astype(np.float64)
            convert = True
        else: 
            q = q.astype(np.float64)
            convert = False

        if len(q.shape) > 1:
            x = []
            for iq in q: 
                x.append(self.model.param_constrain(iq)*1.)
            x = np.array(x)
        else:
            x = self.model.param_constrain(q)

        if convert: x = tf.constant(x, dtype=tf.float32)
        return x


    def unconstrain_samples(self, q):
        
        if type(q) != np.ndarray:
            q = q.numpy().astype(np.float64)
            convert = True
        else: 
            q = q.astype(np.float64)
            convert = False

        if len(q.shape) > 1:
            x = []
            for iq in q: 
                x.append(self.model.param_unconstrain(iq)*1.)
            x = np.array(x)
        else:
            x = self.model.param_unconstrain(q)

        if convert: x = tf.constant(x, dtype=tf.float32)
        return x




if __name__=="__main__":

    try: 
        model_n = int(sys.argv[1])
    except Exception as e:
        print("Exception while parsing command line argument for model number : ", e)
        print("\nRequires a model number as a command line input")
        sys.exit()

    print("Run for model number %d "%model_n)
    
    model, refsamples = setup_pdb_model(model_n)

    #
    D = model.dims()
    q = np.random.normal(1, 0.3, D)

    print()
    print(q)
    print("log_density and gradient of the model:")
    print(model.log_density_gradient(q))
    print(model.log_density_gradient(q, propto = 1, jacobian = 0))
    print()

    # print("Start sampling with HMC")
    # stepsize = 0.001
    # steps = 100
    # metric_diag = np.ones(D)
    # nsamples = 10
    # #
    # #SET random seed to replicate sampling
    # np.random.seed(2)
    # sampler = mcmc.HMCDiag(model, stepsize=stepsize, steps=steps, metric_diag=metric_diag)
    # theta = np.empty([nsamples, model.dims()])
    # for m in range(nsamples):
    #     theta[m, :], _ = sampler.sample()

    # print(theta)


