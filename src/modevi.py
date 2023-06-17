import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import polyak



@tf.function
def evidence_loss_grad(model, log_likelihood, sample1, sample2, log_prior):
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        logl1 = tf.stop_gradient(log_likelihood(sample1))
        logpr1 = log_prior(sample1)
        logp1 = logl1 + logpr1

        logl2 = tf.stop_gradient(log_likelihood(sample2))
        logpr2 = log_prior(sample2)
        logp2 = logl2 + logpr2

        logq1 = model.log_prob(sample1)
        logq2 = model.log_prob(sample2)
        f = (logq1 - logp1) - (logq2 - logp2)
        f = tf.reduce_mean(f)
        elbo =  -1. * tf.reduce_mean(logq1 - logp1)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return f, elbo, gradients



@tf.function
def grad_project_norm(model, log_likelihood, log_prior, sample):

    sample = tf.stop_gradient(sample)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)

        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = log_likelihood(sample)
            logpr = log_prior(sample)
            logp = logl + logpr
            logq = model.log_prob(sample)
            elbo = tf.reduce_mean(logq - logp, axis=0)

        grad_p = tape_in.gradient(logp, sample)
        grad_q = tape_in.gradient(logq, sample)

        dotpq = tf.einsum('ij, ij -> i', grad_p, grad_q)
        #dotqq = tf.einsum('ij, ij -> i', grad_q, grad_q)
        dotpp = tf.einsum('ij, ij -> i', grad_p, grad_p)
        f = tf.reduce_mean(dotpq - dotpp)
        
    gradients = tape.gradient(f, model.trainable_variables)
    return elbo, f, gradients



@tf.function
def hill_climb(sample, log_likelihood, log_prior):
    print('Hill climb')
    with tf.GradientTape(persistent=True) as tape_in:
        tape_in.watch(sample)
        logl = log_likelihood(sample)
        logpr = log_prior(sample)
        logp = logl + logpr
        f = -logp
        f = tf.reduce_mean(f)
    grad_sample = tape_in.gradient(f, sample)
    return f, grad_sample

    

#@tf.function
def modevi(model, log_likelihood, log_prior, advsample, opt_h, opt_q, sample_dist, phase, batch=tf.constant(32), nsteps=100, early_stop=True, threshold=1e-3, patience=20):   
    if phase == 1: sample = sample_dist(batch)
    elif phase == 2 : sample = model.sample(batch)
    advsample.assign(sample)
    #mean = np.array([77.51446,   11.813156,   2.9884362])
    print("sample begin : \n", np.array(advsample))
    #print("dist begin : " , tf.reduce_sum(mean - advsample)**2)
    #reset optimizer
    for var in opt_h.variables():
        var.assign(tf.zeros_like(var))

    losses = []
    for j in range(nsteps):
        f, grad_sample = hill_climb(advsample, log_likelihood, log_prior)
        losses.append(f)
        opt_h.apply_gradients(zip([grad_sample], [advsample]))

        if j > 10:
            elbo, f_gnorm, gradients = grad_project_norm(model, log_likelihood, log_prior, advsample)
            #opt_q.apply_gradients(zip(gradients, model.trainable_variables))
            opt_q.prepare(f, gradients)
            opt_q.apply_gradients(zip(gradients, model.trainable_variables))
            
        if early_stop:
            if early_stopping(losses, threshold, patience, j) : break
            else: pass                
    #print(advsample)
    print("sample end : \n", np.array(advsample))
    print()
    return elbo, f_gnorm




def early_stopping(losses, threshold, patience, epoch):
    if (len(losses) > patience):
        lossmax, lossmin = max(losses[-patience:]), min(losses[-patience:])
        change = (lossmax-lossmin)/lossmax
        if change < threshold:
            print("Loss has decreased by less than %0.2f percentage in last %d iterations"%(threshold*100, patience))
            print("Stop iteration at iteration %d"%epoch)
            return True
    else:
        return False


    
def check_hill_climb(x0, log_likelihood, prior=False, lr=1e-3, niter=1001, nprint=None, verbose=True, callback=None, threshold=1e-3, early_stop=True, patience=20):
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if nprint is None: nprint = niter //10
    if not prior: log_prior = lambda x : 0.
    x = tf.Variable(x0)
    losses = []
    for epoch in range(niter+1):
        
        loss, grads = hill_climb(x, log_likelihood, log_prior)
        opt.apply_gradients(zip([grads], [x]))
        losses.append(loss)
        if (epoch %nprint == 0) & verbose: 
            print("Loss at epoch %d is"%epoch, loss)
            print("gradnorm : ", (tf.linalg.global_norm([grads])).numpy())

        if early_stop:
            if early_stopping(losses, threshold, patience, epoch) : break
            else: pass                
    return x, np.array(losses)



def train(qdist, log_likelihood, sample_dist, phase,
          prior=False, batch=1, niter=1001,
          lr_adv=1e-3, nsteps_adv=10,
          nprint=None, verbose=True, callback=None,
          **kwargs):

    print(kwargs)
    kw = kwargs

    if nprint is None: nprint = niter //10
    if not prior: log_prior = lambda x : 0.
    opt_q = polyak.Polyak(gamma=kw['gamma'], beta=kw['beta'], epsmax=kw['epsmax'])

    #Initialize optimizers for adverserial sampling
    advsample = tf.Variable(qdist.sample(batch)*1)
    opt_h = tf.keras.optimizers.Adam(learning_rate=lr_adv)
    _, grads = hill_climb(advsample, log_likelihood, log_prior)
    opt_h.apply_gradients(zip([grads], [advsample]))

    #
    #MAIN OPTIMIZATION LOOP
    elbos, epss, losses = [], [], []
    for epoch in range(niter+1):

        elbo, loss = modevi(qdist, log_likelihood, log_prior=log_prior,
                            advsample=advsample, phase=phase,
                            opt_h=opt_h, opt_q=opt_q, sample_dist=sample_dist, nsteps=nsteps_adv,
                            batch=tf.constant(batch),
                            )
        #
        elbo, loss = elbo.numpy(), loss.numpy()
        epss.append(opt_q.eps)
        losses.append(loss)
        elbos.append(elbo)
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            print("Eps, loss, gradnorm are : ", opt_q.eps, loss, opt_q.gradnorm)
            if callback is not None:
                callback(qdist, [np.array(losses), np.array(elbos), np.array(epss)], epoch + niter*(phase-1))

    return qdist, np.array(losses), np.array(elbos), np.array(epss)



