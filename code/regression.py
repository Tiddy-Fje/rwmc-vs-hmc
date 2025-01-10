from utils import test_example, plot_acf, preprocess_data, logposterior, plot_eff_sample_size
from sampler import RandomWalkMCMC, HamiltonianMCMC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd

X,y,sigmas = preprocess_data()
logdensity, grad_logdensity = logposterior(X,y,sigmas)

mass = 0.05
leapfrog_time = 0.06
dt = 0.02

sample_hamiltonian = HamiltonianMCMC(0, np.zeros(X.shape[1]), logdensity, 0, grad_logdensity, np.ones(X.shape[1])*mass, leapfrog_time, dt)
result = sample_hamiltonian.sample(1,1000)

keys = ['age', 'weight', 'african/american', 'other race', 'smoke', 'premature birth',
                'hypertension', 'uterine irritability', 'first physician visit', 'more visits', 'intercept']   

for i in range(result.shape[2]):
    plt.plot(result[0,:,i], label=keys[i])
    plt.plot(np.mean(result[0,:,i]))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('t')
plt.grid()
plt.savefig('../figures/traceplot_hamiltonian_regression.png', bbox_inches='tight')

sample_hamiltonian = HamiltonianMCMC(0, np.zeros(X.shape[1]), logdensity, 200, grad_logdensity, np.ones(X.shape[1])*mass, leapfrog_time, dt)
result = sample_hamiltonian.sample(1,5000)

fig = plt.figure(figsize=(10,4))
df = pd.melt(pd.DataFrame(result[0,:,:], columns=keys), var_name='covariates', value_name='samples')
plot = sns.violinplot(data=df, x='covariates', y='samples', inner='quart', width=1)
plt.xlim(-1, 11)
plot.set(xlabel=None, ylabel=None)
plt.xticks(fontsize=10, rotation=45)
plt.grid()
plt.savefig('../figures/violin_hamiltonian_regression.png', bbox_inches='tight')

mean_HMC = np.mean(result[0,:,:], axis=0)

sample_hamiltonian = RandomWalkMCMC(3, np.zeros(X.shape[1]), logdensity, 200, 0.1)
result_RW = sample_hamiltonian.sample(1,35000)
mean_RW = np.mean(result_RW[0,:,:], axis=0)

clf = LogisticRegression().fit(X[:,:-1], y)
regression = list(clf.coef_[0])
regression.append(clf.intercept_[0])

results = {'HMC': mean_HMC, 'Logistic Regression': regression, 'RW': mean_RW}

x = np.arange(11)
width = 0.28
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(10)
for attribute, measurement in results.items():
    measurement = np.round(measurement, 2)
    offset = width * multiplier
    rects = ax.bar(x + offset - width*1.5, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3, fontsize=7)
    multiplier += 1

plt.xticks(ticks=x, labels=keys, rotation=45)
plt.legend()
plt.savefig('../figures/regression_comparison.png', bbox_inches='tight')