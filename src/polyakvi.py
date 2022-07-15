import numpy as np
import tensorflow as tf 



@tf.function
def polyakvi(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        sample1 = model.sample(nsamples)
        logl1 = log_likelihood(sample1)
        if prior is None:
            logpr1 = 0.
        else:
            logpr1 = model.log_prior(sample1)
        logp1 = logl1 + logpr1
        logq1 = model.log_prob(sample1)

        sample2 = model.sample(nsamples)
        logl2 = log_likelihood(sample2)
        if prior is None:
            logpr2 = 0.
        else:
            logpr2 = model.log_prior(sample2)
        logp2 = logl2 + logpr2
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients



@tf.function
def polyakvi_score(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   

    sample1 = tf.stop_gradient(model.sample(nsamples))
    logl1 = tf.stop_gradient(log_likelihood(sample1))
    if prior is None:
        logpr1 = 0.
    else:
        logpr1 = model.log_prior(sample1)
    logp1 = logl1 + logpr1

    sample2 = tf.stop_gradient(model.sample(nsamples))
    logl2 = tf.stop_gradient(log_likelihood(sample2))
    if prior is None:
        logpr2 = 0.
    else:
        logpr2 = model.log_prior(sample2)
    logp2 = logl2 + logpr2

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq1 = model.log_prob(sample1)
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients



@tf.function
def polyakvi_qgrad(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        sample1 = model.sample(nsamples)
        logl1 = tf.stop_gradient(log_likelihood(sample1))
        if prior is None:
            logpr1 = 0.
        else:
            logpr1 = model.log_prior(sample1)
        logp1 = logl1 + logpr1

        sample2 = model.sample(nsamples)
        logl2 = tf.stop_gradient(log_likelihood(sample2))
        if prior is None:
            logpr2 = 0.
        else:
            logpr2 = model.log_prior(sample2)
        logp2 = logl2 + logpr2

        logq1 = model.log_prob(sample1)
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients


def train(model, log_likelihood, beta=0., gamma=0.9, mode='full', epsmax=0., nsamples=1, niter=1001, nprint=100, verbose=True):

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=beta)
    elbos, epss, losses = [], [], []
    for epoch in range(niter):
        
        if mode == 'full': elbo, loss, grads = polyakvi(model, log_likelihood, nsamples=tf.constant(nsamples))
        elif mode == 'score': elbo, loss, grads = polyakvi_score(model, log_likelihood, nsamples=tf.constant(nsamples))
        elif mode == 'qonly': elbo, loss, grads = polyakvi_qgrad(model, log_likelihood, nsamples=tf.constant(nsamples))
        elbo, loss = elbo.numpy(), loss.numpy()
        #
        gradnorm = (tf.linalg.global_norm(grads)).numpy()
        eps = gamma*loss/gradnorm**2
        if epsmax > 0:
            eps = min(epsmax, abs(eps))
            eps *= np.sign(loss)
        if np.isnan(eps):
            print("NaNs!!! :: ", epoch, eps, loss, gradnorm)
            break

        opt.learning_rate = eps
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epss.append(eps)
        losses.append(loss)
        elbos.append(elbo)
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            print("Eps, loss, gradnorm are : ", eps, loss, gradnorm)
            
    return model, np.array(losses), np.array(elbos), np.array(epss)
