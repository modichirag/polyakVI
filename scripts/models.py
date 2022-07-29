import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()



class PDB_5():
    
    def __init__(self):
        posterior = my_pdb.posterior(pos[5])
        posname = posterior.name
        refdrawsdict = posterior.reference_draws()
        stanmodel, data = posterior.model, posterior.data.values()
        refinfo = posterior.reference_draws_info()
        refdrawsdict = posterior.reference_draws()
        keys = refdrawsdict[0].keys()
        stansamples = []
        for key in keys:
            stansamples.append(np.array([refdrawsdict[i][key] for i in range(len(refdrawsdict))]).flatten())
        #print(stanmodel.code('stan'))
        
        ##Transform
        self.D = 4
        self.offset = 5
        self.earn = tf.constant(np.log(data['earn']).astype(np.float32))
        self.y = self.earn
        self.height = tf.constant(np.log(data['height']).astype(np.float32))
        self.male = tf.constant(np.array(data['male']).astype(np.float32))
        self.samples = np.array(stansamples).copy().astype(np.float32).T

    @tf.function
    def loglik(self, w):
        w0, w1, w2, logsigma = tf.split(w, self.D, 1)
        mu = w0 + w1*self.height + w2*self.male + self.offset
        #sigma = tf.exp(logsigma)
        sigma = tf.nn.softplus(logsigma)
        ll = tfd.Normal(mu, sigma).log_prob(self.y)
        return tf.reduce_sum(ll, axis=1)

    def transform_samples(self, samples):
        samples[:, 0] += self.offset
        samples[:, -1] = tf.nn.softplus(samples[:, -1])
        #samples[-1] = tf.exp(samples[-1])
        return samples

    def gensamples(self, qdist, n=1000):
        samples = qdist.sample(n).numpy()
        return self.transform_samples(samples)


class PDB_1():
    
    def __init__(self):
        posterior = my_pdb.posterior(pos[1])
        posname = posterior.name
        refdrawsdict = posterior.reference_draws()
        stanmodel, data = posterior.model, posterior.data.values()
        refinfo = posterior.reference_draws_info()
        refdrawsdict = posterior.reference_draws()
        keys = refdrawsdict[0].keys()
        stansamples = []
        for key in keys:
            stansamples.append(np.array([refdrawsdict[i][key] for i in range(len(refdrawsdict))]).flatten())
        #print(stanmodel.code('stan'))
        
        ##Transform
        self.D = 5
        self.offset = 88
        self.hs = tf.constant(np.array(data['mom_hs'])- np.mean(data['mom_hs']), dtype=tf.float32)
        self.iq = tf.constant(np.array(data['mom_iq'])- np.mean(data['mom_iq']), dtype=tf.float32)
        self.inter = self.hs*self.iq
        self.y = tf.constant(np.array(data['kid_score'], dtype=np.float32))
        self.samples = np.array(stansamples).copy().astype(np.float32).T

    @tf.function
    def loglik(self, w):
        w0, w1, w2, w3, logsigma = tf.split(w, self.D, 1)
        mu = w0 + w1*self.hs + w2*self.iq + w3*self.inter + self.offset
        sigma = tf.nn.softplus(logsigma)
        ll = tfd.Normal(mu, sigma).log_prob(self.y)
        return tf.reduce_sum(ll, axis=1)


    def transform_samples(self, samples):
        samples[:, -1] = tf.nn.softplus(samples[:, -1])
        samples[:, 0] += self.offset
        return samples

    def gensamples(self, qdist, n=1000):
        samples = qdist.sample(n).numpy()
        return self.transform_samples(samples)




