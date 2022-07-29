import numpy as np
import tensorflow as tf 


@tf.function
def bbvi_score(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   
    sample = tf.stop_gradient((model.sample(nsamples)))
    #sample = (model.sample(nsamples))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logl = log_likelihood(sample)
        if prior: 
            logp = model.log_prior(sample)
        else:
            logp = 0 
        logq = model.log_prob(sample)    
        elbo = tf.stop_gradient(logl + logp - logq)
        loss = -1.* logq* elbo
        #logq = -1. * logq
    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.reduce_mean(elbo), gradients



@tf.function
def bbvi_path(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   
    eps = tf.stop_gradient(model.noise.sample(nsamples))
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.forward(eps)
        logl = log_likelihood(sample)
        if prior: 
            logp = model.log_prior(sample)
        else:
            logp = 0 
        logq = model.log_prob(sample)
        elbo = logl + logp - logq
        negelbo = tf.reduce_mean(-1. * elbo, axis=0)
    gradz = tape.gradient(negelbo, sample)
    gradients = tape.gradient(sample, model.trainable_variables, gradz)
    return tf.reduce_mean(elbo), gradients


@tf.function
def bbvi_elbo(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.sample(nsamples) 
        logl = log_likelihood(sample)
        if prior: 
            logp = model.log_prior(sample)
        else:
            logp = 0 
        logq = model.log_prob(sample)
        elbo = logl + logp - logq
        negelbo = -1. * elbo
    gradients = tape.gradient(negelbo, model.trainable_variables)
    return tf.reduce_mean(elbo), gradients



def train(qdist, log_likelihood, prior=False, opt=None, lr=1e-3, mode='full', nsamples=32, niter=1001, nprint=None, verbose=True, callback=None):
    
    if opt is None: opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elbos, epss, losses = [], [], []
    if nprint is None: nprint = niter //10
    for epoch in range(niter):
        
        if mode == 'full': elbo, grads = bbvi_elbo(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'score': elbo,  grads = bbvi_score(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'path': elbo, grads = bbvi_path(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elbo = elbo.numpy()
        #
        if np.isnan(elbo):
            print("NaNs!!! :: ", epoch, elbo)
            break

        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        elbos.append(elbo)
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            if callback is not None: 
                callback(qdist, [np.array(elbos)], epoch)
    return qdist, np.array(elbos)

