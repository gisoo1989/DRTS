import numpy as np
from scipy.stats import norm
import sobol_seq ## For quasi-Monte carlo estimation

## For quick update of Vinv
def sherman_morrison(X, V, w=1):
    result = V-(w*np.einsum('ij,j,k,kl -> il', V, X, X, V))/(1.+w*np.einsum('i,ij,j ->', X, V, X))
    return result

'''
Linear Thompson sampling algorithm (TS)
'''
class standardTS:
    def __init__(self, d, v):
        ## Initialization
        self.beta_hat=np.zeros(d)
        self.f=np.zeros(d)
        self.Binv=np.eye(d)
        self.t = 0

        ## Hyperparameters
        self.v=v
        self.settings = {'v': self.v}

    def select_ac(self,contexts):
        ## Sample beta_tilde.
        N=len(contexts)
        V=(self.v**2)*self.Binv
        beta_tilde=np.random.multivariate_normal(self.beta_hat, V, size=N)
        est=np.array([np.dot(contexts[i], beta_tilde[i,]) for i in range(N)])
        ## Selecting action with tie-breaking.
        self.action=np.random.choice(np.where(est == est.max())[0])
        self.contexts=contexts
        return(self.action)

    def update(self,reward):
        newX=self.contexts[self.action]
        self.f=self.f+reward*newX
        self.Binv = sherman_morrison(X=newX, V=self.Binv)
        self.beta_hat=np.dot(self.Binv, self.f)




'''
Balanced Linear Thompson sampling algorithm (BLTS)
'''
class BalancedLTS:
    def __init__(self, d, v, gamma):
        ## Initialization
        self.beta_hat=np.zeros(d)
        self.y=np.zeros(d)
        self.Binv=np.eye(d)
        self.Wresidual=0. # weighted residual

        ## Hyperparameters: cross-validating lambda is not used due to online-algorthm scnarios
        self.v=v
        self.gamma = gamma
        self.settings = {'v': self.v, 'gamma':self.gamma}

    def select_ac(self,contexts):
        #context: list with length N
        N = len(contexts)
        means = np.array([np.dot(Xi,self.beta_hat) for Xi in contexts])
        if self.Wresidual==0:
            self.action=np.random.choice(np.arange(0,N,1))
            self.prop_score = np.max([1/N, self.gamma])
        else:
            V = self.v**2*self.Wresidual*self.Binv
            beta_tilde=np.random.multivariate_normal(self.beta_hat, V, size=N)
            est=np.array([np.dot(contexts[i], beta_tilde[i,]) for i in range(N)])
            self.action=np.random.choice(np.where(est == est.max())[0])
            ## Calculate the propensity score
            a_t = self.action
            M=200
            norms = np.array([np.sqrt(X_i.dot(V).dot(X_i)) for X_i in self.contexts])
            j_index = np.hstack((np.arange(0,N,1)[:a_t], np.arange(0,N,1)[(a_t+1):]))
            mean_js = np.array(means[j_index])
            norm_js = np.array(norms[j_index])
            Sobol_vector = sobol_seq.i4_sobol_generate_std_normal(1, M).reshape(-1)*norms[a_t]+means[a_t]
            Pm = norm.cdf(np.subtract.outer( Sobol_vector, mean_js )/norm_js[None,:])
            self.prob_score = np.max([np.mean(np.prod(Pm, axis=1)), self.gamma])
        self.contexts=contexts
        return(self.action)

    def update(self,reward):
        ## Calculate the estimators
        newX=self.contexts[self.action]
        self.y=self.y+(1/self.prop_score)*reward*newX
        self.Binv = sherman_morrison(X=newX, V=self.Binv, w=1/self.prop_score)
        self.beta_hat=(self.Binv).dot(self.y)
        self.Wresidual = self.Wresidual + (1/self.prop_score)*(reward - np.dot(self.beta_hat,newX))**2



'''
Doubly robust Thompson Sampling
'''
class DRTS:
    def __init__(self, d, gamma, v,max_resamp):
        ##Initialization
        self.t=0
        self.d=d
        self.beta_hat=np.zeros(d)
        self.beta_hat2=np.zeros(d)
        self.yx = np.zeros(d)
        self.yx2 = np.zeros(d)
        self.V = np.zeros((d,d))
        self.Vinv2 = np.eye(d)
        self.Vinv = np.eye(d)
        self.max_resamp=max_resamp

        ## Hyperparameters
        self.v=v
        self.gamma = gamma
        self.settings = {'v': self.v, 'gamma':self.gamma}

    def calculate_pi(self,N,M,means,norms,V):
        pi_hats=[]
        seq_std_norm=sobol_seq.i4_sobol_generate_std_normal(1, M).reshape(-1)
        for a in range(N):
            js_index=np.hstack((np.arange(0,N,1)[:a], np.arange(0,N,1)[(a+1):]))
            mean_js = np.array(means[js_index])
            norm_js = np.array(norms[js_index])
            Sobol_vector = seq_std_norm*norms[a]+means[a]
            Pm = norm.cdf(np.subtract.outer( Sobol_vector, mean_js )/norm_js[None,:])
            pi_a = np.mean(np.prod(Pm, axis=1))
            pi_hats.append(pi_a)
        pi_hats=np.array(pi_hats)
        pi_sumovergammas=np.sum(pi_hats[(pi_hats>self.gamma)])
        return([pi_hats,pi_sumovergammas])


    def select_ac(self, contexts):
        ### Calculate the std of normal random variables and sample beta_tilde
        N = len(contexts)
        self.t = self.t + 1
        means = np.array([np.dot(Xi, self.beta_hat) for Xi in contexts])
        if self.t==1:
            a_t = np.random.choice(N)
            self.pi_hat = 1/N
        else:

            V = self.v**2*self.Vinv
            ## Estimation of pi
            M = 200
            norms = np.array([np.sqrt(X_i.dot(V).dot(X_i)) for X_i in self.contexts])

            pi_hats,pi_sumovergammas=self.calculate_pi(N,M,means,norms,V)

            action = 0
            resamp_num =0
            while(resamp_num<=self.max_resamp and action == 0):
                resamp_num+=1

                ## Thompson Sampling
                beta_tilde=np.random.multivariate_normal(self.beta_hat, V, size=N)
                est=np.array([np.dot(contexts[i], beta_tilde[i,]) for i in range(N)])
                a_t=np.random.choice(np.where(est == est.max())[0])



                if pi_hats[a_t] > self.gamma:
                    action = 1
                    self.pi_hat = pi_hats[a_t]/pi_sumovergammas 
            if action==0:
                pi_hats[a_t]=pi_hats[a_t]*(1-pi_sumovergammas)**(self.max_resamp-1)
        ### Save data for update.
        self.action = a_t
        self.contexts = contexts
        self.means = means
        return(self.action)

    def update(self, reward):
        N = len(self.contexts)
        X = np.array(self.contexts)
        DR_Y = X.dot(self.beta_hat2)
        DR_Y[self.action] = DR_Y[self.action] + (1/self.pi_hat)*(reward-self.contexts[self.action].dot(self.beta_hat2))
        self.yx = self.yx + X.T.dot(DR_Y)

        ##Calculate beta_hat
        self.V = self.V + X.T.dot(X)
        self.Vinv = np.linalg.inv(self.V + np.sqrt(self.t)*np.eye(self.d))
        self.beta_hat = (self.Vinv).dot(self.yx)

        newX=self.contexts[self.action]
        self.yx2=self.yx2+newX*reward
        self.Vinv2=sherman_morrison(X=newX, V=self.Vinv2)
        self.beta_hat2=(self.Vinv2).dot(self.yx2)
