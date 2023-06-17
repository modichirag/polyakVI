import numpy as np
import tensorflow as tf 
import divergences as divs
import sys, os


@tf.function
def bbvi_score(qdist, model, batch=tf.constant(32)):   

    sample = tf.stop_gradient((qdist.sample(batch))) *1.
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph bbvi_score")
        tape.watch(qdist.trainable_variables)
        logl = model.log_likelihood_and_grad(sample)
        logq = qdist.log_prob(sample)    
        elbo = tf.stop_gradient(logl - logq)
        loss = -1.* tf.reduce_mean(logq* elbo, axis=0)
    gradients = tape.gradient(loss, qdist.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients



@tf.function
def bbvi_path(qdist, model, batch=tf.constant(32)):   

    eps = tf.stop_gradient(qdist.noise.sample(batch))
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph bbvi_path")
        tape.watch(qdist.trainable_variables)
        sample = qdist.forward(eps)
        logl = model.log_likelihood_and_grad(sample)
        logq = qdist.log_prob(sample)
        elbo = logl - logq
        loss = tf.reduce_mean(-1. * elbo, axis=0) #loss = negelbo
    gradz = tape.gradient(loss, sample)
    gradients = tape.gradient(sample, qdist.trainable_variables, gradz)
    return tf.reduce_mean(elbo), loss , gradients




@tf.function
def bbvi_scorenorm(qdist, model, batch=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''
    sample = tf.stop_gradient(qdist.sample(batch))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(qdist.trainable_variables)
        print("creating graph bbvi_scorenorm")
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = model.log_likelihood_and_grad(sample)
            logp = logl
            logq = qdist.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_theta = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_theta)

    gradients = tape.gradient(loss, qdist.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients


@tf.function
def bbvi_fullgradnorm(qdist, model, batch=tf.constant(32)):   
    '''Implement L_2norm[\grad_theta \log(q) - \log(p)] as the loss
    '''

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(qdist.trainable_variables)
        sample = qdist.sample(batch) * 1.
        print("creating graph bbvi_fullgradnorm")
        
        with tf.GradientTape(persistent=True) as tape_in:
            tape_in.watch(sample)

            logl = model.log_likelihood_and_grad(sample)
            logp = logl
            logq = qdist.log_prob(sample)
            f = logq - logp 
            f = tf.reduce_mean(f)
            elbo =  -1*f  

        grad_theta = tape_in.gradient(f, sample)
        loss = tf.linalg.norm(grad_theta)

    gradients = tape.gradient(loss, qdist.trainable_variables)
    return elbo, loss, gradients



@tf.function
def bbvi_elbo(qdist, model, batch=tf.constant(32)):   

    with tf.GradientTape(persistent=True) as tape:
        print("creating graph bbvi_elbo/full")
        tape.watch(qdist.trainable_variables)
        sample = qdist.sample(batch) 
        try:
            logl = model.log_likelihood_and_grad(sample)
        except:
            logl = model.log_prob(sample)
        logq = qdist.log_prob(sample)
        elbo = logl - logq
        negelbo = -1. * tf.reduce_mean(elbo, axis=0)
        loss = negelbo
        
    gradients = tape.gradient(negelbo, qdist.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients


# @tf.function
# def qdivergence(qdist, model, nsamples=32):
#     print("create graph qdivergence")
#     samples = qdist.sample(nsamples)
#     logp1 = qdist.log_prob(samples)
#     logp2 = model.log_likelihood_and_grad(samples)
#     div = tf.reduce_mean(logp1 - logp2)
#     return div, tf.reduce_mean(logp1), tf.reduce_mean(logp2)


# @tf.function
# def fkl_divergence(samples, qdist, model=None):
#     print("create graph fkl_divergence")
#     logp1 = qdist.log_prob(samples)
#     if model is not None:
#         logp2 = model.log_likelihood_and_grad(samples)
#     else:
#         logp2 = 0.
#     div = tf.reduce_mean(logp1 - logp2)
#     return div, tf.reduce_mean(logp1), tf.reduce_mean(logp2)


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



def train(qdist, model, lr=1e-3, mode='full', batch=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None, nqsamples=1000, savepath=None):

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    val_and_grad_func = parse_mode(mode)
    elbos, losses, qdivs, counts, fdivs = [], [], [], [], []

    for epoch in range(niter+1):

        #compare
        qdiv = divs.qdivergence(qdist, model, nsamples=nqsamples)
        qdivs.append(qdiv)
        if samples is not None:
            fdiv = divs.fkl_divergence(qdist, model, samples=samples)
            fdivs.append(fdiv)
        else:
            fdivs = None

        #grad and update
        elbo, loss, grads = val_and_grad_func(qdist, model, batch=tf.constant(batch))
        elbo = elbo.numpy()
        if np.isnan(elbo):
            print("ELBO is NaNs :: ", epoch, elbo)
            break

        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        elbos.append(elbo)
        losses.append(loss)
        counts.append(model.grad_count) 
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            sys.stdout.flush()
            if callback is not None: callback(qdist, [np.array(elbos)], epoch)
            if savepath is not None:
                np.save(savepath + 'elbo', np.array(elbos))
                np.save(savepath + 'qdivs', np.array(qdivs))
                np.save(savepath + 'losses', np.array(losses))
                np.save(savepath + 'counts', np.array(counts))
                np.save(savepath + 'fdivs', np.array(fdivs))
                qsamples = qdist.sample(1000)
                qsamples = model.constrain_samples(qsamples).numpy()
                np.save(savepath + 'samples', qsamples)


    print("return")
    return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)
    

def train_frg(qdist, model, lr=1e-3, mode='full', batch=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None, nqsamples=1000, savepath=None, ntries=10):

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    val_and_grad_func = parse_mode(mode)
    elbos, losses, qdivs, counts, fdivs = [], [], [], [], []

    for epoch in range(niter+1):

        #compare
        #measure divergences
        qdiv = divs.qdivergence(qdist, model, nsamples=nqsamples)
        fdiv = divs.fkl_divergence(qdist, model, samples=samples)
        qdivs.append(qdiv)
        fdivs.append(fdiv)
            
        #grad and update
        loc, scale = qdist.loc*1., qdist.scale*1.
        run = 1
        for itry in range(ntries):
            try:
                elbo, loss, grads = val_and_grad_func(qdist, model, batch=tf.constant(batch))
                if ~np.isnan(elbo) :
                    run = 0
            except Exception as e:
                print(f"Iteration {epoch}, try {itry}, exception in iteration for : \n" , e)
            if run == 0 :
                break
        if itry > ntries:
            print("Max iterations reached")
            raise Exception
        
        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        try:
            sample = qdist.sample()
        except Exception as e:
            print('Exception in sampling after update, revert to previous point')
            qdist.loc.assign(loc)
            qdist.scale.assign(scale)
            print()

        #Logging
        elbos.append(elbo.numpy())
        losses.append(loss)
        counts.append(model.grad_count) 
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            sys.stdout.flush()
            if callback is not None: callback(qdist, [np.array(elbos)], epoch)
            if savepath is not None:
                np.save(savepath + 'elbo', np.array(elbos))
                np.save(savepath + 'qdivs', np.array(qdivs))
                np.save(savepath + 'losses', np.array(losses))
                np.save(savepath + 'counts', np.array(counts))
                np.save(savepath + 'fdivs', np.array(fdivs))
                qsamples = qdist.sample(1000)
                qsamples = model.constrain_samples(qsamples).numpy()
                np.save(savepath + 'samples', qsamples)


    print("return")
    return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)
    
