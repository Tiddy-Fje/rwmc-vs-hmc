import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pandas as pd

def two_dim_input_fun( fun, grid_x, grid_y ):
    arg = [grid_x,grid_y]
    result = fun(np.array(arg).reshape(2, -1))
    return result.reshape( grid_x.shape )

def test_example( alpha, beta=0.25, plot=False ):
    '''
    Test function for the unnormalized density.
    '''
    def unnorm_logdensity( x ):
        return -alpha * ( (x[0]**2 + x[1]**2) - beta ) ** 2
    def unnorm_logdensity_grad( x ):
        return -4 * alpha * ( (x[0]**2 + x[1]**2) - beta ) * x
    
    if plot:
        x = np.linspace(-1., 1., 100)
        y = np.linspace(-1., 1., 100)
        X, Y = np.meshgrid(x, y)
        fig, ax = plt.subplots()
        Z = np.exp(unnorm_logdensity([X,Y]))
        ax.contourf(X, Y, Z/np.sum(Z), levels=200)
        ax.set_xlabel('q_1')
        ax.set_ylabel('q_2')
        plt.savefig(f'../figures/density_alpha={alpha}.png')
    
    return unnorm_logdensity, unnorm_logdensity_grad

def preprocess_data():
    '''
    Preprocesses the birthwt dataset:
    - Standardizes the age and weight columns.
    - Transform race and physician visit columns into binary columns.
    '''
    df = pd.read_csv('../data/birthwt.csv')
    
    df = df.rename(columns={'ptl': 'premature birth', 'ht': 'hypertension', 'ui': 'uterine irritability'})

    df['age'] = (df['age'] - df['age'].mean())/df['age'].std()
    df['weight'] = (df['lwt'] - df['lwt'].mean())/df['lwt'].std()

    df['african/american'] = df['race'].apply(lambda x: 1 if x == 2 else 0)
    df['other race'] = df['race'].apply(lambda x: 1 if x == 3 else 0)
    df['first physician visit'] = df['ftv'].apply(lambda x: 1 if x > 0 else 0)
    df['more physician visits'] = df['ftv'].apply(lambda x: 1 if x > 1 else 0)

    df['intercept'] = 1

    keys = ['low', 'age', 'weight', 'african/american', 'other race', 'smoke', 'premature birth', 'hypertension',
                'uterine irritability', 'first physician visit', 'more physician visits', 'intercept']    
    df = df[keys]
    keys.remove('low')

    log_odds_dic = {key: np.exp(1) for key in keys}
    #log_odds_dic['premature birth'] *= 5.
    #log_odds_dic['weight'] *= 5.
    log_odds = [log_odds_dic[key] for key in keys]
    sigmas = np.log( np.array(log_odds) )*10
    
    X = np.array(df.drop(columns = ['low']))
    y = np.array(df['low'])
    return X, y, sigmas

def logposterior(X, y, sigma):
    temp = X.T @ (y - 1)
    def logdensity( q ):
        #print(np.min(X@q))
        return q.T @ temp - np.sum(np.log(1 + np.exp(- X @ q))) - 0.5 * np.sum(q**2/sigma**2)
    def logdensity_grad( q ):
        return temp + X.T @ (1/(1 + np.exp(X @ q))) - q/sigma**2
    return logdensity, logdensity_grad

def plot_2d_kde( samples1, samples2, title1, title2, figname ):
    '''
    Plots the 2D KDE for a given array of shape (n_chains, n_samples, 2).
    Parameters:
    samples (np.ndarray): An array of shape (n_chains, n_samples, 2).
    '''
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    def sub_plot(samples, title, ax):
        sns.kdeplot(
            x=samples[0,:, 0], 
            y=samples[0,:, 1], 
            fill=True, 
            alpha=0.4, 
            ax=ax,
        )
        
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    sub_plot(samples1, title1, ax[0])
    sub_plot(samples2, title2, ax[1])

    plt.tight_layout()
    plt.savefig( f'../figures/{figname}.png' )

    return

if __name__ == '__main__':
    print('Testing stuff ...')

def plot_eff_sample_size(samples1, samples2, label1, label2, dt, t):
    def plot_RWMC(samples, label):
        acf_q1 = np.array([acf(samples[i,:,0], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)
        acf_q2 = np.array([acf(samples[i,:,1], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)
        ns = [int(n) for n in np.linspace(len(acf_q1)/50, len(acf_q1)/5, 20)]
        eff_q1 = [n/(1+2*sum(acf_q1[1:n])) for n in ns]
        eff_q2 = [n/(1+2*sum(acf_q2[1:n])) for n in ns]

        plt.plot(ns, eff_q1, label=label1+' - q1')
        plt.plot(ns, eff_q2, label=label1+' - q2')
    def plot_HMC(samples, label, dt, t):
        acf_q1 = np.array([acf(samples[i,:,0], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)
        acf_q2 = np.array([acf(samples[i,:,1], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)
        ns = [int(n) for n in np.linspace(len(acf_q1)/500, len(acf_q1)/50, 20)]
        eff_q1 = [n/(1+2*sum(acf_q1[1:n])) for n in ns]
        eff_q2 = [n/(1+2*sum(acf_q2[1:n])) for n in ns]
        ns = [n*(1+2*t/dt) for n in ns]
        plt.plot(ns, eff_q1, label=label2+' - q1')
        plt.plot(ns, eff_q2, label=label2+' - q2')
    
    if label1 == 'RWMC': plot_RWMC(samples1, label1)
    else: plot_HMC(samples1, label1, dt, t)
    if label2 == 'RWMC': plot_RWMC(samples2, label2)
    else: plot_HMC(samples2, label2, dt, t)
    plt.legend()
    plt.ylabel('Effective sample size')
    plt.xlabel('Number of evaluations')
    plt.grid()
    #plt.savefig( f'../figures/effective_sample_size.png' )
    plt.show()

def plot_acf(samples1, samples2, label1, label2):
    for i in range(2):
        samples = samples1 if i == 0 else samples2
        label = label1 if i == 0 else label2
        acf_q1 = np.array([acf(samples[i,:,0], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)
        acf_q2 = np.array([acf(samples[i,:,1], nlags=len(samples[0])) for i in range(len(samples))]).mean(axis=0)

        plt.plot(acf_q1[:200], label=label+' - q1')
        plt.plot(acf_q2[:200], label=label+' - q2')
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid()
    plt.savefig( f'../figures/acf.png' )
    plt.show()