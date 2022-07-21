import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class MF_Gaussian(tf.Module):


    def __init__(self, d, mu=0, scale=1, name=None):
        super(MF_Gaussian, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.zeros(shape=[self.d]) + mu, name='loc')
        self.std = tf.Variable(tf.ones(shape=[self.d]) * scale, name='std') 
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))      
    
    
    @property
    def q(self):
        """Variational distribution"""
        #return tfd.Normal(self.loc, tf.nn.softplus(self.std))
        return tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=tf.nn.softplus(self.std))
        #return tfd.Normal(self.loc, tf.exp(self.std))


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        x = self.loc + z*tf.nn.softplus(self.std)
        #x = self.loc + z*tf.exp(self.std)
        return x

    def inverse(self, x):
        z = (x - self.loc)/tf.nn.softplus(self.std)
        #z = (x - self.loc)/tf.exp(self.std)
        return z




class FR_Gaussian(tf.Module):


    def __init__(self, d, mu=0, scale=1, name=None):
        super(FR_Gaussian, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.zeros(shape=[self.d]) + mu, name='loc')
        self.scale = tfp.util.TransformedVariable(
            tf.eye(d, dtype=tf.float32) *scale, tfp.bijectors.FillScaleTriL(),
            name="rascale_tril")
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))      
    
    
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

    def forward(self, z):
        #x = self.loc + self.scale @ z
        #x = self.loc + tf.matmul(self.scale , z)
        x = self.loc + tf.matmul(z, self.scale)
        return x

    def inverse(self, x):
        xm = (x - self.loc)
        #z = tf.linalg.inv(self.scale)@xm
        #z = tf.matmul(tf.linalg.inv(self.scale), xm)
        z = tf.matmul(xm, tf.linalg.inv(self.scale))
        return z


