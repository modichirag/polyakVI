import numpy as np
import tensorflow as tf 
import os, sys
sys.path.append('../../hmc/src/')
from pyhmc import PyHMC


def get_grad_log_prob(q, log_likelihood):   

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        loglik = log_likelihood(q)
    gradients = tape.gradient(loglik, q)
    return gradients




def sample(log_prob, grad_log_prob, step_size=0.1, Nleapfrog=None, nsamples=100, burnin=100, q0=None, D=None, nchains=1):


    log_prob_np = lambda x: log_prob(tf.constant(x.reshape(1, -1), dtype=tf.float32)).numpy()[0]
    if grad_log_prob is None:
        grad_log_prob_np = lambda x: get_grad_log_prob(tf.constant(x.reshape(1, -1), dtype=tf.float32), log_prob).numpy()[0]
    else:
        grad_log_prob_np = lambda x: grad_log_prob(tf.constant(x.reshape(1, -1), dtype=tf.float32)).numpy()[0]


    hmc = PyHMC(log_prob_np, grad_log_prob_np)

    if Nleapfrog is None: Nleapfrog = int(np.pi/2/step_size)

    stepfunc =  lambda x:  hmc.hmc_step(x, Nleapfrog, step_size)

    if q0 is None: q0 = np.random.normal(0, 1, D).reshape(1, -1)
    if len(q0.shape) == 1: q0 = q0.reshape(1, -1)
    #
    samples = []
    accepts = []
    probsi = []
    countsi = []

    q = q0.copy()
    print("sample : ", q)
    print(q.shape)
    print(log_prob_np(q), q)
    print(grad_log_prob_np(q))    
    
    print("Starting HMC loop")

    #q = [q*np.random.normal(1, 0.01, 1) for _ in range(nchains)]
    q = [q for _ in range(nchains)]
    for i in range(nsamples+burnin):
        out = list(map(stepfunc, q))
        q = [i[0] for i in out]
        acc = [i[2] for i in out]
        prob = [i[3] for i in out]
        count = [i[4] for i in out]
        #q, mom, acc, prob, count = stepfunc(q)
        samples.append(q)
        accepts.append(acc)
        probsi.append(prob)
        countsi.append(count)
        if i%100 == 0: 
            print("Iteration %d"%i)
            acc_fraction = (np.array(accepts)==1).sum()/(np.array(accepts)).size
            print("Accepted = %0.3f"%acc_fraction)

    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probsi)[burnin:]
    counts = np.array(countsi)[burnin:]

    return mysamples, accepted, probs, counts


