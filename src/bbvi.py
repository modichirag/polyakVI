import numpy as np
import tensorflow as tf 


@tf.function
def bbvi(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   
    sample = model.sample(nsamples)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logl = log_likelihood(sample)
        if prior is None:
            logp = 0.
        else:
            logp = model.log_prior(sample)
        logq = model.log_prob(sample)    
        elbo = tf.stop_gradient(logl + logp - logq)
        loss = -1.* logq* elbo
        logq = -1. * logq
    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.reduce_mean(elbo), gradients



@tf.function
def bbvi_reparam(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   
    eps = model.noise.sample(nsamples)
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.forward(eps)
        logl = log_likelihood(sample)
        if prior is None:
            logp = 0.
        else:
            logp = model.log_prior(sample)
        logq = model.log_prob(sample)
        elbo = logl + logp - logq
        negelbo = tf.reduce_mean(-1. * elbo, axis=0)
    gradz = tape.gradient(negelbo, sample)
    gradients = tape.gradient(sample, model.trainable_variables, gradz)
    return tf.reduce_mean(elbo), gradients


@tf.function
def bbvi_elbo(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.sample(nsamples) 
        logl = log_likelihood(sample)
        if prior is None:
            logp = 0.
        else:
            logp = model.log_prior(sample)
        logq = model.log_prob(sample)
        elbo = logl + logp - logq
        negelbo = -1. * elbo
    gradients = tape.gradient(negelbo, model.trainable_variables)
    return tf.reduce_mean(elbo), gradients


@tf.function
def train(model, optimizer, likelihood, niter):
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.reduce_mean(elbo)

