import numpy as np
import tensorflow as tf 



class Polyak():
    
    def __init__(self, gamma, beta=0., epsmax=0.):
        
        self.gamma = gamma
        self.beta = beta 
        self.eps_init = 0.1
        self.epsmax = epsmax
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.eps_init, momentum=self.beta)
        self.gradnorm = None
        self.prepared = False
    
    def prepare(self, loss, grads):
        self.gradnorm = (tf.linalg.global_norm(grads)).numpy()
        eps = self.gamma*abs(loss)/self.gradnorm**2
        if self.epsmax > 0.:
            eps = min(self.epsmax, eps)
            eps = eps * np.sign(loss)
        self.eps = eps
        self.opt.learning_rate = self.eps
        self.prepared = True
    
    def apply_gradients(self, zipped_grads_and_vars):
        if self.prepared:
            self.opt.apply_gradients(zipped_grads_and_vars)
            self.prepared = False
        else: 
            print("Need to prepare stepsize before update")
            raise RuntimeError


class Polyak_slack():
    
    def __init__(self, llambda, delta, beta=0.):
        
        self.llambda = llambda
        self.delta = delta
        self.beta = beta 
        self.eps_init = 0.1
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.eps_init, momentum=self.beta)
        self.slack = None
        self.prepared = False

    def prepare(self, loss, grads):

        self.gradnorm = (tf.linalg.global_norm(grads)).numpy()
        #if self.slack is None: self.slack = abs(loss)
        if self.slack is None: self.slack = 0.

        delta_step =  max(0, abs(loss) - self.slack + self.delta) / (self.delta + self.llambda* self.gradnorm**2)
        eps0 = self.llambda * delta_step
        eps1 = abs(loss)/self.gradnorm**2
        eps = min(eps0, eps1)
        eps = eps * np.sign(loss)
        self.eps = eps
        self.opt.learning_rate = self.eps
        self.slack = max(0, self.slack - self.delta + self.delta * delta_step)
        self.prepare = True

    def apply_gradients(self, zipped_grads_and_vars):
        if self.prepared:
            self.opt.apply_gradients(zipped_grads_and_vars)
            self.prepared = False
        else: 
            print("Need to prepare stepsize before update")
            raise RuntimeError

