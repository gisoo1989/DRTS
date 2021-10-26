import matplotlib.pyplot as plt
import numpy as np
import sys, os, json
import argparse

parser = argparse.ArgumentParser(description='Plot for DRTS set (N, d)')

parser.add_argument('-N', type=int, help='Number of arms', default=10)
parser.add_argument('-d', type=int, help='Dimension of contexts', default=10)
args = parser.parse_args()
N = args.N
d = args.d

with open('./TS_d%d_N%d.txt'% (d, N), 'r') as infile:
    results_TS = json.loads(infile.read())

with open('./BLTS_d%d_N%d.txt'% (d, N), 'r') as infile:
    results_BLTS = json.loads(infile.read())

with open('./DRTS_d%d_N%d.txt'% (d, N), 'r') as infile:
    results_DRTS_lamt = json.loads(infile.read())

###### TS plots ########
RT_TS=[]
#Find best Hyperparameter
for i in range(4):
    regrets = np.array(results_TS[i]['regrets'])
    avg_regrets = np.mean(regrets, axis=0)
    RT_TS.append(avg_regrets[-1])
T = avg_regrets.shape[0]


###### BLTS plots ########
models=['v=0.001, gam=0.01','v=0.001, gam=0.05','v=0.001, gam=0.1', #v=0.001
        'v=0.01, gam=0.01','v=0.01, gam=0.05','v=0.01, gam=0.1',    #v=0.01
        'v=0.1, gam=0.01','v=0.1, gam=0.05','v=0.1, gam=0.1',       #v=0.1
        'v=1, gam=0.01','v=1, gam=0.05','v=1, gam=0.1']             #v=1
RT_BLTS=[]
#Find best Hyperparameter
for i,model in enumerate(models):
    regrets = np.array(results_BLTS[i]['regrets'])
    avg_regrets = np.mean(regrets, axis=0)
    RT_BLTS.append(avg_regrets[-1])
T = avg_regrets.shape[0]


###### DRTS plots ########
models=['v=0.001', #v=0.001
        'v=0.01',    #v=0.01
        'v=0.1',       #v=0.1
        'v=1']             #v=1
RT_DRTS =[]
#Find best Hyperparameter
for i,model in enumerate(models):
    regrets = np.array(results_DRTS_lamt[i]['regrets'])
    avg_regrets = np.mean(regrets, axis=0)
    RT_DRTS.append(avg_regrets[-1])
T = avg_regrets.shape[0]


#list of indexes of best result for each algorithm
indexes=[RT_TS.index(min(RT_TS)), RT_BLTS.index(min(RT_BLTS)), RT_DRTS.index(min(RT_DRTS))]
models=['TS','BLTS','DRTS_lamt']
linestyles=['dotted', 'dashed','solid']
labels=['LinTS','BLTS','DRTS']
colors = ['blue', 'red', 'black', 'green']
plt.figure()
for i, model in enumerate(models):
    exec('regrets=np.array(results_'+model+'['+str(indexes[i])+'][\'regrets\'])')
    avg_regrets = np.mean(regrets, axis=0)
    T = avg_regrets.shape[0]
    plt.plot(np.arange(1,T+1), avg_regrets, linestyle=linestyles[i], label=labels[i], color=colors[i])
#plt.title()
plt.legend(title=('N=%d, d=%d' % (N,d)))
plt.ylabel('Regrets')
plt.xlabel(r'$t$ : Number of Rounds')
plot_dir = './regrets_d%d_N%d.png' % (d,N)
plt.savefig(plot_dir, bbox_inches='tight', pad_inches=0, dpi=200)
plt.close()
print('Saved at ' + plot_dir)


## Beta_err plot
print('Comparing estimation error...')
colors = ['blue', 'red', 'black', 'green']
linestyles=['dotted', 'dashed','solid']
models=['TS','BLTS','DRTS_lamt']
plt.figure()
for i, model in enumerate(models):
    exec('errs=np.array(results_'+model+'['+str(indexes[i])+'][\'beta_err\'])')
    avg_err = np.mean(errs, axis=0)
    std_err = np.std(errs, axis=0)
    T = avg_regrets.shape[0]
    plt.plot(np.arange(1,T+1), avg_err, linestyle=linestyles[i], label=labels[i], color=colors[i])
    plt.fill_between(np.arange(1,T+1), avg_err-std_err, avg_err+std_err, color = colors[i], alpha=0.1)
#plt.title()
plt.legend(title=('N=%d, d=%d' % (N,d)))
plt.ylabel(r'$\|| \beta - \widehat{\beta}\||_{2}$')
plt.xlabel(r'$t$ : Number of Rounds')
plot_dir = './beta_err_d%d_N%d.png' % (d,N)
plt.savefig(plot_dir, bbox_inches='tight', pad_inches=0, dpi=200)
plt.close()
print('Saved at ' + plot_dir)
