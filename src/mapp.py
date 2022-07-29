import numpy as np
import tensorflow as tf


@tf.function
def val_and_grad(sample, log_likelihood, prior=False):   
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(sample)
        logl = log_likelihood(sample)
        if prior: 
            logp = model.log_prior(sample)
        else:
            logp = 0
        logp = logl + logp
        loss = -1. * tf.reduce_mean(logp)

    gradients = tape.gradient(loss, sample)
    return loss, gradients



def train(x0, log_likelihood, prior=False, opt=None, lr=1e-3, niter=1001, nprint=None, verbose=True, callback=None):
    

    if opt is None: opt = tf.keras.optimizers.Adam(learning_rate=lr)
    losses = []
    if nprint is None: nprint = niter //10

    if prior is not None: prior = 0 

    for epoch in range(niter):
        
        loss, grads = val_and_grad(x0, log_likelihood, prior=prior)
        #
        if np.isnan(loss):
            print("NaNs!!! :: ", epoch, loss)
            break

        opt.apply_gradients(zip([grads], [x0]))
        #opt.apply_gradients(grads, x0)
        losses.append(loss)
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, loss)
    return x0, np.array(losses)

