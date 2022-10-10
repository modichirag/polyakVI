##Stan gradients are only implemented for bbvi_full. 
##They are not yet implemented for bbvi_score and bbvi_path
import numpy as np
import tensorflow as tf 


@tf.function
def bbvi_score(model, log_likelihood, grad_log_likelihood=None, prior=False, nsamples=tf.constant(32)):   
    if grad_log_likelihood is not None:
        raise NotImplementedError

    sample = tf.stop_gradient((model.sample(nsamples))) *1.
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
def bbvi_path(model, log_likelihood, grad_log_likelihood=None, prior=False, nsamples=tf.constant(32)):   
    if grad_log_likelihood is not None:
        raise NotImplementedError

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
def _bbvi_elbo(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.sample(nsamples) *1.
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


@tf.function
def _bbvi_elbo_gradloglik(model, log_likelihood, grad_log_likelihood, prior=False, nsamples=tf.constant(32)):   
    with tf.GradientTape(persistent=True) as tape:
        #print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.sample(nsamples) *1.
        #
        if prior: 
            logp = model.log_prior(sample)
        else:
            logp = 0 
        logq = model.log_prob(sample)
        lpq = logp - logq
        neglpq = -1.* lpq

    #logl = log_likelihood(sample)
    logl = tf.numpy_function(log_likelihood, [sample], tf.float32)
    elbo = logl + logp - logq
    negelbo = -1. * elbo

    
    gradients_lpq = tape.gradient(neglpq, model.trainable_variables)
    #gradients_loglik = grad_log_likelihood(sample)
    gradients_loglik = tf.numpy_function(grad_log_likelihood, [sample], tf.float32)
    gradients_loglik = tape.gradient(sample, model.trainable_variables, gradients_loglik)
    gradients = [gradients_lpq[i] - gradients_loglik[i] for i in range(len(gradients_lpq))]
    return tf.reduce_mean(elbo), gradients



def bbvi_elbo(model, log_likelihood, grad_log_likelihood=None, prior=False, nsamples=tf.constant(32)):   
    
    if grad_log_likelihood is None:
        return _bbvi_elbo(model, log_likelihood, prior=prior, nsamples=nsamples)
    else:
        return _bbvi_elbo_gradloglik(model, log_likelihood, grad_log_likelihood, prior=prior, nsamples=nsamples)



def train(qdist, log_likelihood, grad_log_likelihood=None, prior=False, opt=None, lr=1e-3, mode='full', nsamples=32, niter=1001, nprint=None, verbose=True, callback=None):
    
    if opt is None: opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elbos, epss, losses = [], [], []
    if nprint is None: nprint = niter //10
    for epoch in range(niter):
        
        if mode == 'full': elbo, grads = bbvi_elbo(qdist, log_likelihood, grad_log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'score': elbo,  grads = bbvi_score(qdist, log_likelihood, grad_log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'path': elbo, grads = bbvi_path(qdist, log_likelihood, grad_log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
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

