import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def corner(samples, savepath="./tmp/", savename=None, save=True):
    '''Make corner plot for the distribution from samples
    '''
    D = samples.shape[1]
    fig, ax = plt.subplots(D, D, figsize=(3*D, 2*D), sharex='col')

    for i in range(D):
        for j in range(D):
            if i==j:
                ax[i, j].hist(samples[:, i])
                ax[i, j].set_title('W[{}]'.format(i))
            elif i>j:
                ax[i, j].plot(samples[:, j], samples[:, i], '.')
            else:
                ax[i, j].axis('off')

    plt.tight_layout()

    if save:
        if savename is None: savename='corner'
        plt.savefig(savepath + savename)
        plt.close()
    else: return fig, ax



def compare_hist(samples, ref_samples, nbins=20, lbls="", savepath="./tmp/", savename=None, suptitle=None):
    '''Compare histogram of samples with reference sample
    '''
    D = ref_samples.shape[1]
    if (lbls == "") & (type(samples) == list): lbls = [""]*len(samples)

    fig, ax = plt.subplots(1, D, figsize=(3*D, 3))

    for ii in range(D):
        ax[ii].hist(ref_samples[:, ii], bins=nbins, density=True, alpha=1, lw=2, histtype='step', color='k', label='Ref');
        
        if type(samples) == list: 
            for j, ss in enumerate(samples):
                ax[ii].hist(ss[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls[j]);
        else:
            ax[ii].hist(samples[:, ii], bins=nbins, density=True, alpha=0.5, label=lbls);

    ax[0].legend()
    plt.suptitle(suptitle)
    plt.tight_layout()

    if savename is None: savename='compare_hist'
    plt.savefig(savepath + savename)
    plt.close()
    #return fig, ax


def plot_polyak_losses(losses, elbos, epss, savepath="./tmp/", savename=None, suptitle=None, skip=100):

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    
    ax[0].plot(-elbos[skip:])
    ax[0].semilogy()
    ax[0].set_title('-ELBO')
    ax[1].plot(epss[skip:])
    ax[1].set_yscale('symlog', linthresh=1e-5)
    ax[1].set_title('step size')
    ax[2].plot(abs(losses[skip:]))
    ax[2].semilogy()
    ax[2].set_title('Loss')
    
    for axis in ax: axis.grid(which='both', lw=0.5)
    plt.suptitle(suptitle)
    if savename is None: savename='losses'
    plt.savefig(savepath + savename)
    plt.close()
    #return fig, ax
    
def plot_bbvi_losses(elbos, savepath="./tmp/", savename=None, suptitle=None, skip=100):

    plt.plot(-elbos[skip:])
    plt.semilogy()
    plt.title('-ELBO')
    plt.grid(which='both', lw=0.5)
    plt.suptitle(suptitle)
    if savename is None: savename='losses'
    plt.savefig(savepath + savename)
    plt.close()
    
