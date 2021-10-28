import numpy as np
import sys, os, json
from tqdm import trange
from algorithms import standardTS, BalancedLTS, DRTS
import matplotlib.pyplot as plt
import argparse

'''
Import settings from cmd
'''
parser = argparse.ArgumentParser(description='Simulation for DRTS. Set (T, N, d, seed, repeat)')

parser.add_argument('-T', type=int, help='Number of total rounds', default=20000)
parser.add_argument('-N', type=int, help='Number of arms', default=10)
parser.add_argument('-d', type=int, help='Dimension of contexts', default=10)
parser.add_argument('-seed', type=int, help='Random seeds', default=0)
parser.add_argument('-repeat', type=int, help='Number of repetitions', default=10)
args = parser.parse_args()
N = args.N
d = args.d
T = args.T
seed = args.seed
M = args.repeat




##Simulation settings
# standard deviation of error
R=1
#Correlation in contexts
rho = 0.5
if N % 2 == 1:
    X_mean = np.hstack((np.arange(-N+1,0,2),0,np.arange(2,N+1,2)))
else:
    X_mean = np.hstack((np.arange(-N,0,2),np.arange(2,N+1,2)))
X_cov = (1-rho)*np.eye(N) + rho*np.ones((N,N))
# Hyperparameter set
v_set = [0.001, 0.01, 0.1, 1]

# True beta
np.random.seed(seed) #For reproduciblility
beta = np.random.uniform(-1/np.sqrt(d),1/np.sqrt(d),d)

#Models
models=['TS','BLTS_gam01','BLTS_gam05','BLTS_gam1','DRTS_lamt']


'''
Simulation_CPU_TS
'''
results_TS = []
results_BLTS = []
results_DRTS_lamt = []

for v in v_set:
    np.random.seed(seed) #For reproduciblility and fair comparison
    ##Lists

    for model in models:
        exec('cumulated_regret_'+model+'=np.zeros((M,T))')
        exec('beta_err_'+model+'=np.zeros((M,T))')

    for m in range(M):
        print("v= %.3f Simulation %d" % (v, m+1))

        for model in models:
            exec('RWD_'+model+'=list()')

        optRWD=list()

        ##Model
        exec('M_TS=standardTS(d=d, v=' + str(v) + ')'  )
        exec('M_BLTS_gam01=BalancedLTS(d=d, v=' + str(v) +', gamma=0.01)'  )
        exec('M_BLTS_gam05=BalancedLTS(d=d, v=' + str(v) + ', gamma=0.05)'  )
        exec('M_BLTS_gam1=BalancedLTS(d=d, v=' + str(v) + ', gamma=0.1)'  )
        exec('M_DRTS_lamt=DRTS(d=d, v=' + str(v) + ', gamma='+str(1/(N+1))+', max_resamp='+200+')'  )

        for t in trange(T):
            ## Generate contexts
            X_set = np.array([np.random.multivariate_normal(mean=X_mean, cov=X_cov) for i in range(d)])
            u = np.random.uniform(0,1,N)
            contexts = [u[i]*X_set[:,i]/np.linalg.norm(X_set[:,i]) for i in range(N)]

            ## Optimal mean reward
            optRWD.append(np.amax(np.dot(contexts,beta)))
            ## Standard Gaussian error
            err = R*np.random.randn()


            for model in models:
                exec('a_t=M_'+model+'.select_ac(contexts)')
                rwd=np.dot(contexts[a_t],beta)+err
                exec('RWD_'+model+'.append(np.dot(contexts[a_t],beta))')
                exec('M_'+model+'.update(rwd)')
                exec('beta_err_'+model+'[m,t] = np.linalg.norm(M_'+model+'.beta_hat-beta)')

        for model in models:
            exec('cumulated_regret_'+model+'[m,:] = np.cumsum(optRWD)-np.cumsum(RWD_'+model+')')

    #Save results for each v in v_set

    for model in models:
        if model=='TS':
            exec('results_TS.append({\'model\':\''+str(model)+'\', \'settings\' : M_'+model+'.settings,'+
             '\'regrets\' : cumulated_regret_'+model+'.tolist(),'+
             '\'beta_err\' : beta_err_'+model+'.tolist()})')
        elif model in ['BLTS_gam01','BLTS_gam05','BLTS_gam1']:
            exec('results_BLTS.append({\'model\':\''+str(model)+'\', \'settings\' : M_'+model+'.settings,'+
             '\'regrets\' : cumulated_regret_'+model+'.tolist(),'+
             '\'beta_err\' : beta_err_'+model+'.tolist()})')
        elif model=='DRTS_lamt':
            exec('results_DRTS_lamt.append({\'model\':\''+str(model)+'\', \'settings\' : M_'+model+'.settings,'+
             '\'regrets\' : cumulated_regret_'+model+'.tolist(),'+
             '\'beta_err\' : beta_err_'+model+'.tolist()})')

##Save to txt file
with open('./TS_d%d_N%d.txt' % (d, N), 'w') as outfile:
    json.dump(results_TS, outfile)

with open('./BLTS_d%d_N%d.txt' % (d, N), 'w') as outfile:
    json.dump(results_BLTS, outfile)

with open('./DRTS_d%d_N%d.txt' % (d, N), 'w') as outfile:
    json.dump(results_DRTS_lamt, outfile)
