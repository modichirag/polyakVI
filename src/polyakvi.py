import numpy as np
import tensorflow as tf 
import polyak


@tf.function
def polyakvi(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        sample1 = model.sample(nsamples)
        logl1 = log_likelihood(sample1)
        if prior: 
            logpr1 = model.log_prior(sample1)
        else:
            logpr1 = 0 
        logp1 = logl1 + logpr1
        logq1 = model.log_prob(sample1)

        sample2 = model.sample(nsamples)
        logl2 = log_likelihood(sample2)
        if prior: 
            logpr2 = model.log_prior(sample2)
        else:
            logpr2 = 0 
        logp2 = logl2 + logpr2
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        #elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients



@tf.function
def polyakvi_score(model, log_likelihood, prior=False, nsamples=tf.constant(32)):   

    sample1 = tf.stop_gradient(model.sample(nsamples))
    logl1 = tf.stop_gradient(log_likelihood(sample1))
    if prior: 
        logpr1 = model.log_prior(sample1)
    else:
        logpr1 = 0 
    logp1 = logl1 + logpr1

    sample2 = tf.stop_gradient(model.sample(nsamples))
    logl2 = tf.stop_gradient(log_likelihood(sample2))
    if prior: 
        logpr2 = model.log_prior(sample2)
    else:
        logpr2 = 0 
    logp2 = logl2 + logpr2

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq1 = model.log_prob(sample1)
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        #elbo =  -1. * tf.reduce_mean((logq1 - logp1) + (logq2 - logp2))
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients



@tf.function
def polyakvi_scorenorm(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''
    sample = tf.stop_gradient(model.sample(nsamples))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            if prior: 
                logpr = model.log_prior(sample)
            else:
                logpr = 0 
            logp = logl + logpr
            logq = model.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_theta = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_theta)**2

    gradients = tape.gradient(loss, model.trainable_variables)
    return elbo, loss, gradients


# @tf.function
# def polyakvi_qgrad(model, log_likelihood, prior=None, nsamples=tf.constant(32)):   

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




def train(qdist, log_likelihood, prior=False,  mode='full', nsamples=1, niter=1001, nprint=None, verbose=True, callback=None, slack=0, **kwargs):

    print(kwargs)
    kw = kwargs

    if not slack:
        opt = polyak.Polyak(gamma=kw['gamma'], beta=kw['beta'], epsmax=kw['epsmax'])
    else:
        opt = polyak.Polyak_slack(llambda=kw['llambda'], delta=kw['delta'])

    elbos, epss, losses = [], [], []
    if nprint is None: nprint = niter //10

    for epoch in range(niter):
        
        if mode == 'full': elbo, loss, grads = polyakvi(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'score': elbo, loss, grads = polyakvi_score(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'qonly': elbo, loss, grads = polyakvi_qgrad(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        elif mode == 'scorenorm': elbo, loss, grads = polyakvi_scorenorm(qdist, log_likelihood, prior=prior, nsamples=tf.constant(nsamples))
        else:
            print("\nERROR : mode should be one of - full, score, qonly, scorenorm\n")
            raise RuntimeError
            return [None]*4
        elbo, loss = elbo.numpy(), loss.numpy()

        opt.prepare(loss, grads)
        if np.isnan(opt.eps):
            print("NaNs!!! :: ", epoch, opt.eps, loss, opt.gradnorm)
            break

        #opt.learning_rate = eps
        opt.apply_gradients(zip(grads, qdist.trainable_variables))

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



