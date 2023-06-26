import numpy as np
import sys, os
import json, time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from posteriordb import PosteriorDatabase
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()

##Setup your posterior DB environment
PDBPATH = os.path.join('/mnt/home/cmodi/Research/Projects/posteriordb/posterior_database/')
PDBPATH_COMPILED = '/mnt/home/cmodi/Research/Projects/posteriordb/compiled_models/'
CMDSTAN = '/mnt/home/cmodi/Research/Projects/cmdstan/'
BRIDGESTAN = '/mnt/home/cmodi/Research/Projects/bridgestan/'
MODELDIR = '/mnt/home/cmodi/Research/Projects/posteriordb/compiled_models/'
#MODELDIR = './tmp/'

#sys.path.append(BRIDGESTAN)
#import PythonClient as pbs
import bridgestan as bs
bs.set_bridgestan_path(BRIDGESTAN)

exceptional_models = [68]

class UntrustedModel(Exception):
    """Exception raised for models where posteriordb samples are not trusted.

    Attributes:
        model_n -- model number
        message -- explanation of the error
    """

    def __init__(self, model_n):
        self.model_n = model_n
        self.exceptionl_models = exceptional_models
        self.message = f"Posteriordb samples for model {self.model_n} are not trusted.\nLoad PDB_local_posterior model instead"
        super().__init__(self.message)

        
class PDB_local_posterior():

    def __init__(self, model_n, modelname=''):
        

        self.model_path = f'{PDBPATH_COMPILED}/PDB_{model_n:02d}/'
        self.id = f"{model_n:02d}"
        self.model = self.code()
        self.data = self.dataset()
        self.get_reference_draws()
        if modelname == '':
            self.name = self.model_name()
        else:
            self.name = modelname
    
    def code(self):
        """
        Returns stan model code in model_path as a string
        """
        
        file_path = f'{self.model_path}/PDB_{self.id}.stan'
        with open(file_path) as f:
            contents = f.read()

        return contents

    
    def model_name(self):
        """
        Returns stan model code in model_path as a string
        """
        
        try:
            file_path = f'{self.model_path}/PDB_{self.id}.metadata.json'
            with open(file_path, "r") as f:
                contents = json.load(f)
            name = contents["name"]
        except Exception as e:
            print("Exception occured in naming model\n", e)
            name = f"Model_{self.id}"
        return name

    
    def dataset(self):
        """
        Reads and returns data from json format in model_path
        """
        file_path = f'{self.model_path}/PDB_{self.id}.data.json'
        with open(file_path, "r") as f:
            contents = json.load(f)

        return contents

    def dataset(self):
        """
        Reads and returns data from json format in model_path
        """
        file_path = f'{self.model_path}/PDB_{self.id}.data.json'
        with open(file_path, "r") as f:
            contents = json.load(f)

        return contents

    def get_reference_draws(self):
        """
        """

        params_file = f'{self.model_path}/PDB_{self.id}.samples.meta'
        with open(params_file, 'r') as f:
            self.parameters = [i for i in f.readline()]
            
        self.reference_draws = np.load(f'{self.model_path}/PDB_{self.id}.samples.npy')
        self.samples = np.concatenate([i for i in self.reference_draws])

        

def get_pdb_model(model_n):

    pdb = PosteriorDatabase(PDBPATH)
    if model_n < len(pdb.posterior_names()):
        modelname = pdb.posterior_names()[model_n]
    else:
        modelname = None
    try:
        if model_n in exceptional_models:
            raise UntrustedModel(model_n)
        posterior = pdb.posterior(modelname)
        posname = posterior.name
        stanmodel, data = posterior.model, posterior.data.values()
        refdrawsdict = posterior.reference_draws()
        keys = refdrawsdict[0].keys()
        stansamples = []
        for key in keys:
            stansamples.append(np.array([refdrawsdict[i][key] for i in range(len(refdrawsdict))]).flatten())
        samples = np.array(stansamples).copy().astype(np.float32).T
        
    except Exception as e:
        print(f"Exception occured in loading model: {e}\nLooking in local models")
        posterior = PDB_local_posterior(model_n, modelname=modelname)
        posname = posterior.name
        stanmodel, data = posterior.model, posterior.data.values()
        samples = posterior.samples.reshape(-1, posterior.samples.shape[-1])
        
    print("Model name :", posname)

    return stanmodel, data, samples


def setup_pdb_model(model_n):

    stanmodel, data, samples = get_pdb_model(model_n)

    #Save stan model code
    modeldir = MODELDIR + '/PDB_%02d/'%model_n
    os.makedirs(modeldir, exist_ok=True)
    modpath = modeldir + 'PDB_%02d'%model_n
    modpath = os.path.abspath(modpath)

    #if not present, save model code and data
    if not os.path.isfile(modpath + '.stan'): 
        with open(modpath + '.stan', 'w') as f:
            f.write(stanmodel.code('stan'))
        
    datapath = modpath + '.data.json'
    if not os.path.isfile(datapath): 
        with open(datapath, 'w') as f:
            json.dump(data, f,)

    #Save compiled shared object
    sopath = modpath + '_model.so'

    start = time.time()
    try:
        #if os.path.isfile(sopath):
        #    print("PDB_%02d_model.so file exists"%model_n)
        #model = pbs.PyBridgeStan(sopath, datapath)
        #model = bs.StanModel(os.path.abspath(modeldir) + '/', datapath)
        model = bs.StanModel(sopath, datapath)
        
    except Exception as e:
        print("Exception occured in loading model.so: ", e)
        print("Coompiling model")
        model = bs.StanModel.from_stan_file(modpath + '.stan', datapath)

    return model, samples


class PDBModel_TF():
    
    def __init__(self, model_n):

        self.model_n = model_n
        self.model, self.samples = setup_pdb_model(model_n)
        self.dims = self.model.param_unc_num()
        self.grad_count = 0 

    @tf.custom_gradient
    def log_likelihood_and_grad(self, q):
        
        def np_loglik_and_grad(x):
            self.grad_count += 1
            lk, lk_g = self.model.log_density_gradient(np.float64(x))
            return np.float32(lk), np.float32(lk_g)

        def loglik_and_grad(x):
            lk, lk_g = tf.numpy_function(np_loglik_and_grad, [tf.cast(x, tf.float64)], Tout=[tf.float32,tf.float32])
            return lk, lk_g

        lk, lk_g = tf.map_fn(loglik_and_grad, q, fn_output_signature=(tf.float32, tf.float32))
        
        def grad(upstream):
            return tf.expand_dims(upstream, -1) * lk_g
        return lk, grad


    def reset_gradcount(self):
        self.grad_count = 0 

        
    # def log_likelihood(self, q):  #THIS IS SLOWER THAN THE LOOP BELOW
        
    #     def np_loglik(x):
    #         lk = self.model.log_density(np.float64(x))
    #         return np.float32(lk)
        
    #     def loglik(x):
    #         lk = tf.numpy_function(np_loglik, [tf.cast(x, tf.float64)], Tout=tf.float32)
    #         return lk
        
    #     lk = tf.map_fn(loglik, q, fn_output_signature=tf.float32)
    #     return lk

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

    def log_prob(self, q):
        return self.log_likelihood(q)
    
        
    # def grad_log_likelihood(self, q):
        
    #     def np_grad_loglik(x):
    #         print(x)
    #         self.grad_count += 1
    #         lk_g = self.model.log_density_gradient(np.float64(x))[1]
    #         print("lkg : ", lk_g)
    #         return np.float32(lk_g)
        
    #     def grad_loglik(x):
    #         print(x)
    #         lk_g = tf.numpy_function(np_grad_loglik, [tf.cast(x, tf.float64)], Tout=tf.float32)
    #         return lk_g
        
    #     lk_g = tf.map_fn(grad_loglik, q, fn_output_signature=tf.float32)
    #     return lk_g

    def grad_log_likelihood(self, q):
        
        if type(q) != np.ndarray:
            q = q.numpy().astype(np.float64)
            convert = True
        else: 
            q = q.astype(np.float64)
            convert = False

        if len(q.shape) > 1:
            lk_g = []
            for iq in q:
                self.grad_count += 1
                lk_g.append(self.model.log_density_gradient(iq)[1].astype(np.float32)*1.)
            lk_g = np.array(lk_g)
        else:
            self.grad_count += 1
            lk_g = self.model.log_density_gradient(q)[1]

        if convert: lk_g = tf.constant(lk_g, dtype=tf.float32)
        return lk_g

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
                #x.append(self.model.param_unconstrain(iq)*1.)
                x.append(self.model.param_unconstrain(iq.flatten())*1.)
            x = np.array(x)
        else:
            x = self.model.param_unconstrain(q)

        if convert: x = tf.constant(x, dtype=tf.float32)
        return x





# class TFModels(tf.Module):

#     def __init__(self, q):
#         self.q = q

#     @property
#     def q(self):
#         """Variational distribution"""
#         return tfd.MultivariateNormalTriL(loc = self.loc, scale_tril = self.scale)


#     def __call__(self, x):
#         return self.log_prob(x)

#     def log_prob(self, x):
#         return self.q.log_prob(x)

#     def sample(self, n=1, sample_shape=None):
#         return self.q.sample(n)

#     @tf.function
#     def log_likelihood(self, q):
#         return self.log_prob(q)


#     @tf.function
#     def grad_log_likelihood(self, q):
#         with tf.GradientTape() as tape:
#             tape.watch(q)
#             lp = self.log_prob(q)
#         grad = tape.gradient(lp, q)
#         return grad

    
#     @tf.custom_gradient
#     def log_likelihood_and_grad(self, q):
#         lp = self.log_likelihood(q)
#         lp_g = self.grad_log_likelihood(q)
#         return lp, lp_g


#     def constrain_samples(self, q):
#         return q

#     def unconstrain_samples(self, q):
#         return q



class FRGuassian_model(tf.Module):


    def __init__(self, d, mu=0, scale=1, name=None):
        super(FRGuassian_model, self).__init__(name=name)
        self.dims = d
        self.loc = tf.Variable(tf.zeros(shape=[self.dims]) + mu, name='loc')
        self.scale = tfp.util.TransformedVariable(
            tf.eye(d, dtype=tf.float32) *scale, tfp.bijectors.FillScaleTriL(),
            name="rascale_tril")
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.dims))      
        self.samples = self.q.sample(10000).numpy()
    
    @property
    def q(self):
        """Variational distribution"""
        return tfd.MultivariateNormalTriL(loc = self.loc, scale_tril = self.scale)


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    @tf.function
    def log_likelihood(self, q):
        return self.log_prob(q)


    @tf.function
    def grad_log_likelihood(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self.log_prob(q)
        grad = tape.gradient(lp, q)
        return grad

    
    @tf.custom_gradient
    def log_likelihood_and_grad(self, q):
        lp = self.log_likelihood(q)
        lp_g = self.grad_log_likelihood(q)
        def grad(upstream):
            return tf.expand_dims(upstream, -1) * lp_g
        return lp, grad


    def constrain_samples(self, q):
        return q

    def unconstrain_samples(self, q):
        return q


    
class MF_Gaussian_mixture_model(tf.Module):


    def __init__(self, n, d, mu=0, scale=1, name=None):
        super(MF_Gaussian_mixture_model, self).__init__(name=name)
        self.n = n
        self.dims = d
        
        self.p = tf.Variable(tf.ones(shape=[self.n])/self.n, name='pi')
        self.loc = tf.Variable(tf.zeros(shape=[self.n, self.dims]) + mu, name='loc')
        self.std = tf.Variable(tf.ones(shape=[self.n, self.dims]) * scale, name='std') 
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.dims))      
        self.samples = self.q.sample(10000).numpy()
    
    
    @property
    def q(self):
        """Variational distribution"""
        q = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=tf.nn.softplus(self.p)),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.loc,
                scale_diag=tf.nn.softplus(self.std)
            )
        )
        return q


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    @tf.function
    def log_likelihood(self, q):
        return self.log_prob(q)


    @tf.function
    def grad_log_likelihood(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self.log_prob(q)
        grad = tape.gradient(lp, q)
        return grad

    
    @tf.custom_gradient
    def log_likelihood_and_grad(self, q):
        lp = self.log_likelihood(q)
        lp_g = self.grad_log_likelihood(q)
        def grad(upstream):
            return tf.expand_dims(upstream, -1) * lp_g
        return lp, grad


    def constrain_samples(self, q):
        return q

    def unconstrain_samples(self, q):
        return q


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
    #D = model.dims()
    D = model.param_num()
    print("Dimensions : ", D)
    uD = model.param_unc_num()
    print("Unconstrained number of dimensions : ", uD)
    
    q = np.random.normal(1, 0.3, D)
    cq = model.param_constrain(q)
    #uq = model.param_unconstrain(q)
    ucq = model.param_unconstrain(model.param_constrain(q))
    print(q.shape, cq.shape, ucq.shape)
    
    print()
    print(q)
    print(cq)
    if D == uD: print(q/cq)
    #print(ucq)
    #print(q/cq)
    print("log_density and gradient of the model:")
    print(model.log_density_gradient(q))
    #print(model.log_density_gradient(uq))
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


