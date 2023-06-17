##Stan gradients are only implemented for bbvi_full. 
##They are not yet implemented for bbvi_score and bbvi_path
import numpy as np
import tensorflow as tf 


@tf.function
def bbvi_score(model, log_likelihood, log_prior, batch=tf.constant(32)):   

    sample = tf.stop_gradient((model.sample(batch))) *1.
    #sample = (model.sample(batch))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logl = log_likelihood(sample)
        logpr = log_prior(sample)
        logq = model.log_prob(sample)    
        elbo = tf.stop_gradient(logl + logpr - logq)
        loss = -1.* tf.reduce_mean(logq* elbo, axis=0)
        #logq = -1. * logq
    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients



@tf.function
def bbvi_path(model, log_likelihood, log_prior, batch=tf.constant(32)):   

    eps = tf.stop_gradient(model.noise.sample(batch))
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.forward(eps)
        logl = log_likelihood(sample)
        logpr = log_prior(sample)
        logq = model.log_prob(sample)
        elbo = logl + logpr - logq
        loss = tf.reduce_mean(-1. * elbo, axis=0) #loss = negelbo
    gradz = tape.gradient(loss, sample)
    gradients = tape.gradient(sample, model.trainable_variables, gradz)
    return tf.reduce_mean(elbo), loss , gradients



@tf.function
def bbvi_evidence(model, log_likelihood, log_prior, batch=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        sample1 = model.sample(batch) * 1.
        logl1 = log_likelihood(sample1)
        logpr1 = log_prior(sample1)
        logp1 = logl1 + logpr1
        logq1 = model.log_prob(sample1)

        sample2 = model.sample(batch) * 1.
        logl2 = log_likelihood(sample2)
        logpr2 = log_prior(sample2)
        logp2 = logl2 + logpr2
        logq2 = model.log_prob(sample2)
        
        f = (logq1 - logp1) - (logq2 - logp2)
        loss = tf.reduce_mean(f**2, axis=0)
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients


@tf.function
def bbvi_scorenorm(model, log_likelihood, log_prior, batch=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''
    sample = tf.stop_gradient(model.sample(batch))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            logpr = log_prior(sample)
            logp = logl + logpr
            logq = model.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_theta = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_theta)

    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients


@tf.function
def bbvi_fullgradnorm(model, log_likelihood, log_prior, batch=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        sample = model.sample(batch) * 1.
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            logpr = log_prior(sample)
            logp = logl + logpr
            logq = model.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_theta = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_theta)

    gradients = tape.gradient(loss, model.trainable_variables)
    return elbo, loss, gradients



@tf.function
def _bbvi_elbo(model, log_likelihood, log_prior, batch=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        print("creating graph")
        tape.watch(model.trainable_variables)
        sample = model.sample(batch) *1.
        logl = log_likelihood(sample)
        if log_prior is not None:
            logpr = log_prior(sample)
        else: logpr = 0.
        logq = model.log_prob(sample)
        elbo = logl + logpr - logq
        negelbo = -1. * tf.reduce_mean(elbo, axis=0)
        loss = negelbo
        
    gradients = tape.gradient(negelbo, model.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients


# @tf.function
# def _bbvi_elbo_gradloglik(model, log_likelihood, grad_log_likelihood, log_prior, batch=tf.constant(32)):
#     #I think this is outdated now
#     with tf.GradientTape(persistent=True) as tape:
#         #print("creating graph")
#         tape.watch(model.trainable_variables)
#         sample = model.sample(batch) *1.
#         #
#         if prior: 
#             logp = model.log_prior(sample)
#         else:
#             logp = 0 
#         logq = model.log_prob(sample)
#         lpq = logp - logq
#         neglpq = -1.* lpq

#     #logl = log_likelihood(sample)
#     logl = tf.numpy_function(log_likelihood, [sample], tf.float32)
#     elbo = logl + logp - logq
#     negelbo = -1. * elbo
#     loss = negelbo

#     gradients_lpq = tape.gradient(neglpq, model.trainable_variables)
#     #gradients_loglik = grad_log_likelihood(sample)
#     gradients_loglik = tf.numpy_function(grad_log_likelihood, [sample], tf.float32)
#     gradients_loglik = tape.gradient(sample, model.trainable_variables, gradients_loglik)
#     gradients = [gradients_lpq[i] - gradients_loglik[i] for i in range(len(gradients_lpq))]
#     return tf.reduce_mean(elbo), loss, gradients


def qdivergence(qdist, model_log_prob, nsamples=32):

    samples = qdist.sample(nsamples)
    logp1 = qdist.log_prob(samples)
    logp2 = model_log_prob(samples)
    div = tf.reduce_mean(logp1 - logp2)
    return div, tf.reduce_mean(logp2), tf.reduce_mean(logp2)


def bbvi_elbo(model, log_likelihood, log_prior, grad_log_likelihood=None, batch=tf.constant(32)):   
    
    if grad_log_likelihood is None:
        return _bbvi_elbo(model, log_likelihood, log_prior=log_prior, batch=batch)
    else:
        print("Outdated code")
        raise Exception
        #return _bbvi_elbo_gradloglik(model, log_likelihood, grad_log_likelihood, prior=prior, batch=batch)



def parse_mode(mode):
    if mode == 'full': val_and_grad_func = bbvi_elbo
    elif mode == 'score': val_and_grad_func = bbvi_score
    elif mode == 'path': val_and_grad_func = bbvi_path
    elif mode == 'evidence': val_and_grad_func = bbvi_evidence
    elif mode == 'scorenorm': val_and_grad_func = bbvi_scorenorm
    elif mode == 'fullgradnorm': val_and_grad_func = bbvi_fullgradnorm
    else:
        print("\nERROR : mode not recognized\n")
        raise RuntimeError
    return val_and_grad_func
    
#######
def train(qdist, log_likelihood, grad_log_likelihood=None, prior=False, opt=None, lr=1e-3, mode='full', batch=32, niter=1001, nprint=None, verbose=True, callback=None):

    print("train function")
    if opt is None: opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if nprint is None: nprint = niter //10
    if not prior: log_prior = lambda x : 0.
    
    val_and_grad_func = parse_mode(mode)
    #
    #MAIN OPTIMIZATION LOOP
    elbos, epss, losses = [], [], []
    losses = []
    for epoch in range(niter+1):

        elbo, loss, grads = val_and_grad_func(qdist, log_likelihood, log_prior=log_prior, batch=tf.constant(batch))
        elbo = elbo.numpy()
        #
        if np.isnan(elbo):
            print("NaNs!!! :: ", epoch, elbo)
            break

        opt.apply_gradients(zip(grads, qdist.trainable_variables))

        elbos.append(elbo)
        losses.append(loss)

        # if prior : model_log_prob = lambda x: log_likelihood(x) + log_prob(x)
        # else: model_log_prob = lambda x: log_likelihood(x)
        # qdiv = qdivergence(qdist, model_log_prob)
        # qdiv.append(qdiv)
        
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            if callback is not None: 
                callback(qdist, [np.array(elbos)], epoch)

    return qdist, np.array(losses), np.array(elbos)


