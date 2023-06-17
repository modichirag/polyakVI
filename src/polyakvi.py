import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import polyak


@tf.function
def polyakvi(model, log_likelihood, log_prior, nsamples=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        sample1 = model.sample(nsamples) * 1.
        logl1 = log_likelihood(sample1)
        logpr1 = log_prior(sample1)
        logp1 = logl1 + logpr1
        logq1 = model.log_prob(sample1)

        sample2 = model.sample(nsamples) * 1.
        logl2 = log_likelihood(sample2)
        logpr2 = log_prior(sample2)
        logp2 = logl2 + logpr2
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients


@tf.function
def _score_grad(model, log_likelihood, sample1, sample2, log_prior):

    logl1 = tf.stop_gradient(log_likelihood(sample1))
    logpr1 = log_prior(sample1)
    logp1 = logl1 + logpr1

    logl2 = tf.stop_gradient(log_likelihood(sample2))
    logpr2 = log_prior(sample2)
    logp2 = logl2 + logpr2
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq1 = model.log_prob(sample1)
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return f, elbo, gradients



@tf.function
def polyakvi_score(model, log_likelihood, log_prior, nsamples=tf.constant(32), sample_dist=None):   
    #print("polyakvi score")
    #if np.random.uniform() > 1.0: sample_dist = tfd.Uniform([0., 0., 1.], [100., 2., 4]).sample
    #else: sample_dist = None
    if sample_dist is None:
        print("sample qdist")
        sample1 = tf.stop_gradient(model.sample(nsamples))
    else:
        print('sample prior')
        sample1 = sample_dist(nsamples)

    #Second sample
    if sample_dist is None:
        sample2 = tf.stop_gradient(model.sample(nsamples))
    else:
        #print('sample prior')
        sample2 = sample_dist(nsamples)
    
    f, elbo, gradients = _score_grad(model, log_likelihood, sample1, sample2, log_prior=log_prior)
    return elbo, f, gradients



@tf.function
def _grad_ascent(model, log_likelihood, log_prior, sample, lpq1):

    with tf.GradientTape(persistent=True) as tape_in:
        tape_in.watch(sample)
        logl = log_likelihood(sample)
        logpr = log_prior(sample)
        logp = logl + logpr
        logq = model.log_prob(sample)
        f = -tf.abs(lpq1 - (logq - logp))
        f = tf.reduce_mean(f)
    grad_sample = tape_in.gradient(f, sample)
    return f, grad_sample

    

#@tf.function
def polyakvi_score_adv(model, log_likelihood, advsample, opt, log_prior, nsamples=tf.constant(32), nsteps=10):   
    '''start from near sample1
    '''

    sample1 = tf.stop_gradient(model.sample(nsamples)*1.+0.)
    logl1 = tf.stop_gradient(log_likelihood(sample1))
    logpr1 = log_prior(sample1)
    logp1 = logl1 + logpr1
    logq1 = model.log_prob(sample1)
    lpq1 = logq1 - logp1

    #Adverserially generate sample2
    advsample.assign(sample1*1.001)

    #reset optimizer
    for var in opt.variables():
        var.assign(tf.zeros_like(var))

    for j in range(nsteps):
        f, grad_sample = _grad_ascent(model, log_likelihood, log_prior, advsample, lpq1)
        #print(advsample, grad_sample)
        opt.apply_gradients(zip([grad_sample], [advsample]))

    sample2 = tf.stop_gradient(advsample*1.)    
    f, elbo, gradients = _score_grad(model, log_likelihood, sample1, sample2, log_prior=log_prior)
    return elbo, f, gradients


#@tf.function
def polyakvi_score_adv2(model, log_likelihood, advsample, opt, log_prior, nsamples=tf.constant(32), nsteps=10):   
    '''generates inedependent 
    '''

    sample1 = tf.stop_gradient(model.sample(nsamples)*1.+0.)
    logl1 = tf.stop_gradient(log_likelihood(sample1))
    logpr1 = log_prior(sample1)
    logp1 = logl1 + logpr1
    logq1 = model.log_prob(sample1)
    lpq1 = logq1 - logp1

    #Adverserially generate sample2
    sample = model.sample(nsamples) * 1.
    advsample.assign(sample*1.)
    
    #reset optimizer
    for var in opt.variables():
        var.assign(tf.zeros_like(var))

    for j in range(nsteps):
        f, grad_sample = _grad_ascent(model, log_likelihood, log_prior, advsample, lpq1)
        opt.apply_gradients(zip([grad_sample], [advsample]))

    sample2 = tf.stop_gradient(advsample*1.)
    f, elbo, gradients = _score_grad(model, log_likelihood, sample1, sample2, log_prior=log_prior)
    return elbo, f, gradients



@tf.function
def _grad_ascent_gradnorm(model, log_likelihood, log_prior, sample):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(sample)

        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            logpr = log_prior(sample)
            logp = logl + logpr
            logq = model.log_prob(sample)
            f = logq - logp
            f = tf.reduce_mean(f)

        grad_sample = tape_in.gradient(f, sample)
        loss = -1. * tf.linalg.norm(grad_sample)

    gradients_adv = tape.gradient(loss, sample)
    return loss, gradients_adv


#@tf.function
def polyakvi_scorenorm_adv(model, log_likelihood, advsample, opt, log_prior, nsamples=tf.constant(32), nsteps=10):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''
    sample = tf.stop_gradient(model.sample(nsamples)) * 1.
    advsample.assign(sample)
    
    #Adverserially generate sample
    #reset optimizer
    for var in opt.variables():
        var.assign(tf.zeros_like(var))

    for j in range(nsteps):
        f, grad_sample = _grad_ascent_gradnorm(model, log_likelihood, log_prior, advsample)
        opt.apply_gradients(zip([grad_sample], [advsample]))
    sample = tf.stop_gradient(advsample*1.)    

    #Now revert to usual scorenorm
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

        grad_sample = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_sample)

    gradients = tape.gradient(loss, model.trainable_variables)
    return elbo, loss, gradients




@tf.function
def polyakvi_scorenorm(model, log_likelihood, log_prior, nsamples=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''
    sample = tf.stop_gradient(model.sample(nsamples))

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

        grad_sample = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_sample)

    gradients = tape.gradient(loss, model.trainable_variables)
    return elbo, loss, gradients


@tf.function
def polyakvi_fullgradnorm(model, log_likelihood, log_prior, nsamples=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        sample = model.sample(nsamples) * 1.
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            logpr = log_prior(sample)
            logp = logl + logpr
            logq = model.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_sample = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_sample)

    gradients = tape.gradient(loss, model.trainable_variables)
    return elbo, loss, gradients


# @tf.function
# def polyakvi_qgrad(model, log_likelihood, log_prior, nsamples=tf.constant(32)):   

#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(model.trainable_variables)

#         sample1 = model.sample(nsamples)
#         logl1 = tf.stop_gradient(log_likelihood(sample1))
#         if prior: 
#             logpr1 = model.log_prior(sample1)
#         else:
#             logpr1 = 0 
#         logp1 = logl1 + logpr1

#         sample2 = model.sample(nsamples)
#         logl2 = tf.stop_gradient(log_likelihood(sample2))
#         if prior: 
#             logpr2 = model.log_prior(sample2)
#         else:
#             logpr2 = 0 
#         logp2 = logl2 + logpr2

#         logq1 = model.log_prob(sample1)
#         logq2 = model.log_prob(sample2)
#         f = (logq1 - logp1) - (logq2 - logp2)
#         f = tf.reduce_mean(f)
#         elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        
#     gradients = tape.gradient(f, model.trainable_variables)
#     return elbo, f, gradients




def train(qdist, log_likelihood, prior=False,  mode='full', nsamples=1, niter=1001, nprint=None, verbose=True, callback=None, slack=0, lr_adv=1e-3, nsteps_adv=10, **kwargs):

    print(kwargs)
    kw = kwargs

    if nprint is None: nprint = niter //10
    if not prior: log_prior = lambda x : 0.
    if not slack:
        opt = polyak.Polyak(gamma=kw['gamma'], beta=kw['beta'], epsmax=kw['epsmax'])
    else:
        opt = polyak.Polyak_slack(llambda=kw['llambda'], delta=kw['delta'])

    #Initialize optimizers for adverserial sampling
    advsample = tf.Variable(qdist.sample(nsamples)*1)
    opt_adv = tf.keras.optimizers.Adam(learning_rate=lr_adv)
    _, grads = _grad_ascent(qdist, log_likelihood, log_prior, advsample, lpq1=qdist.log_prob(advsample))
    opt_adv.apply_gradients(zip([grads], [advsample]))


    #Which mode to work with
    if mode == 'full': val_and_grad_func = polyakvi
    elif mode == 'score': val_and_grad_func = polyakvi_score
    elif mode == 'score_adv': val_and_grad_func = polyakvi_score_adv
    elif mode == 'score_adv2': val_and_grad_func = polyakvi_score_adv2 
    elif mode == 'scorenorm': val_and_grad_func = polyakvi_scorenorm 
    elif mode == 'scorenorm_adv': val_and_grad_func = polyakvi_scorenorm_adv
    elif mode == 'fullgradnorm': val_and_grad_func = polyakvi_fullgradnorm
    else:
        print("\nERROR : mode not recognized\n")
        raise RuntimeError
    
    #
    #MAIN OPTIMIZATION LOOP
    elbos, epss, losses = [], [], []
    for epoch in range(niter+1):

        if "adv" in mode:
            elbo, loss, grads = val_and_grad_func(qdist, log_likelihood, log_prior=log_prior, nsamples=tf.constant(nsamples),
                                                                advsample=advsample, opt=opt_adv, nsteps=nsteps_adv)
        else:
            elbo, loss, grads = val_and_grad_func(qdist, log_likelihood, log_prior=log_prior, nsamples=tf.constant(nsamples))

        opt.prepare(loss, grads)
        opt.apply_gradients(zip(grads, qdist.trainable_variables))

        #
        elbo, loss = elbo.numpy(), loss.numpy()
        epss.append(opt.eps)
        losses.append(loss)
        elbos.append(elbo)
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            print("Eps, loss, gradnorm are : ", opt.eps, loss, opt.gradnorm)
            #print(opt.opt.lr, opt.opt.learning_rate)
            if slack : print("slack : ", opt.slack)
            if callback is not None:
                callback(qdist, [np.array(losses), np.array(elbos), np.array(epss)], epoch)
                
    return qdist, np.array(losses), np.array(elbos), np.array(epss)



