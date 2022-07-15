import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class MF_Gaussian(tf.Module):


    def __init__(self, d, mu=0, scale=1, name=None):
        super(MF_Gaussian, self).__init__(name=name)
        self.d = d
        #self.w_loc = tf.Variable(tf.random.normal(shape=[self.d]), name='w_loc')
        #self.w_std = tf.Variable(tf.random.normal(shape=[self.d]), name='w_std')
        self.w_loc = tf.Variable(tf.zeros(shape=[self.d]) + mu, name='w_loc')
        self.w_std = tf.Variable(tf.ones(shape=[self.d]) * scale, name='w_std') 
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))      
    
    
    @property
    def q(self):
        """Variational distribution"""
        #return tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))
        return tfd.MultivariateNormalDiag(loc=self.w_loc, scale_diag=tf.nn.softplus(self.w_std))
        #return tfd.Normal(self.w_loc, tf.exp(self.w_std))


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        x = self.w_loc + z*tf.nn.softplus(self.w_std)
        #x = self.w_loc + z*tf.exp(self.w_std)
        return x

    def inverse(self, x):
        z = (x - self.w_loc)/tf.nn.softplus(self.w_std)
        #z = (x - self.w_loc)/tf.exp(self.w_std)
        return z


