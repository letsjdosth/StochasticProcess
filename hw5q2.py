import numpy as np
from scipy.stats import poisson as sp_pois
from scipy.stats import gamma as sp_gamma
from scipy.stats import dirichlet as sp_dir
import matplotlib.pyplot as plt

from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_Diag

fetal_lamb_movements = [
    0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,2,2,0,0,0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,7,3,2,3,2,4,0,0,0,0,1,0,0,0,0,0,0,0,1,0,2,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,1,0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0,1,2,0,
    0,0,1,0,1,1,0,1,0,0,2,0,1,2,1,1,2,1,0,1,1,0,0,1,1,0,0,0,1,1,1,0,4,0,0,2,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]

class HW5Q2(MCMC_Gibbs):
    def __init__(self, initial, y_data, hyperparam_dict=None):
        #param
        # 0                          1    2 
        #[[lambda0,lambda1,lambda2], {z}, Q=[q_ij]]
        self.y_data = y_data #np array
        self.n = len(y_data)
        
        self.K = 3 #|S| of {z}
        #hyperparamters
        if hyperparam_dict is None:
            self.hyper_cd_lambda = ((1,3), (2,2), (3,1))
            self.hyper_Aij = np.array([
                [10, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

        else:
            self.hyper_cd_lambda = (
                (hyperparam_dict["hyper_c_lambda_0"], hyperparam_dict["hyper_d_lambda_0"]),
                (hyperparam_dict["hyper_c_lambda_1"], hyperparam_dict["hyper_d_lambda_1"]), 
                (hyperparam_dict["hyper_c_lambda_2"], hyperparam_dict["hyper_d_lambda_2"])
            )
            self.hyper_Aij = hyperparam_dict["hyper_Aij"]

        self.MC_sample = []
        self.MC_sample.append(initial)

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_lambda(new)
        new = self.full_conditional_sampler_z(new)
        new = self.full_conditional_sampler_Q(new)
        self.MC_sample.append(new)

    def full_conditional_sampler_lambda(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0                          1    2 
        #[[lambda0,lambda1,lambda2], {z}, Q=[q_ij]]
        #update new
        count_by_z = [0 for _ in range(self.K)]
        ysum_by_z = [0 for _ in range(self.K)]
        for i, z in enumerate(last_param[1]): #z: 0, 1, ..., k-1
            count_by_z[z] += 1
            ysum_by_z[z] += self.y_data[i]
        new_lambda = [None for _ in range(self.K)]
        for k in range(self.K):
            shape = self.hyper_cd_lambda[k][0] + ysum_by_z[k]
            rate = self.hyper_cd_lambda[k][1] + count_by_z[k]
            new_lambda_k = sp_gamma.rvs(shape, scale=1/rate, size=1)
            new_lambda[k] = new_lambda_k[0]
        new_sample[0] = new_lambda
        return new_sample

    def full_conditional_sampler_z(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0                          1    2 
        #[[lambda0,lambda1,lambda2], {z}, Q=[q_ij]]
        lam = last_param[0]
        Q = np.array(last_param[2])
        #update new
        forward_mat_seq = [np.zeros((self.K, self.K)) for _ in range(2, self.n+1)]
        #index 0 -> F2
        #index (n-2) -> Fn

        #Forward, F2
        f2_vec = np.array([Q[0,s]*
                           sp_pois.pmf(self.y_data[0],lam[0])*sp_pois.pmf(self.y_data[1],lam[s]) 
                           for s in range(self.K)])
        forward_mat_seq[0][0, ] = f2_vec/np.sum(f2_vec)
        #Forward, recursion
        for t in range(3, self.n+1):
            idx = t-2
            for r in range(self.K):
                ft_r_vec = np.zeros((self.K))
                sum_term = np.sum(forward_mat_seq[idx-1][:, r])
                for s in range(self.K):
                    ft_r_vec[s] = Q[r,s] * sp_pois.pmf(self.y_data[t-1], lam[s]) * sum_term
                forward_mat_seq[idx][r, ] = ft_r_vec
            forward_mat_seq[idx] = forward_mat_seq[idx]/np.sum(forward_mat_seq[idx])
        #Backward, z_n
        new_z = [0 for _ in range(self.n)]
        new_z_n_probvec = np.sum(forward_mat_seq[-1], axis=0)
        new_z_n = np.random.choice([k for k in range(self.K)], size=1, p=new_z_n_probvec)
        new_z[-1] = new_z_n[0]
        #Backward, recursion
        for t in [self.n-r for r in range(1,self.n)]:
            z_next_t = new_z[t]
            new_z_idx = t-1
            forward_mat_idx = t-1

            new_z_t_probvec = forward_mat_seq[forward_mat_idx][:, z_next_t].flatten()
            new_z_t = np.random.choice([k for k in range(self.K)], size=1, p=new_z_t_probvec/sum(new_z_t_probvec))
            new_z[new_z_idx] = new_z_t[0]
        new_z[0] = 0 #Z1 is fixed
        new_sample[1] = new_z
        return new_sample

    def full_conditional_sampler_Q(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0                          1    2 
        #[[lambda0,lambda1,lambda2], {z}, Q=[q_ij]]
        #update new
        count_Nij = np.zeros((self.K, self.K))
        z_vec = last_param[1]
        for t in range(1, len(z_vec)):
            count_Nij[z_vec[t-1], z_vec[t]] += 1
        new_Q = np.zeros((self.K, self.K))
        for i in range(self.K):
            dir_param = self.hyper_Aij[i, ] + count_Nij[i, ]
            new_Qi = sp_dir.rvs(dir_param, size=1)
            new_Q[i, ] = new_Qi
        new_sample[2] = new_Q
        return new_sample

if __name__=="__main__":
    np.random.seed(20250321)
    #param
    # 0                          1    2 
    #[[lambda0,lambda1,lambda2], {z}, Q=[q_ij]]
    #Z1 should be 0
    initial = [[0,1,2], [0 for _ in range(len(fetal_lamb_movements))],np.ones((3,3))/3]
    mc_inst = HW5Q2(initial, fetal_lamb_movements)

    mc_inst.generate_samples(5000)
    burnin = 2000
    thinning = 5

    diag_inst_lambda = MCMC_Diag()
    diag_inst_lambda.set_mc_samples_from_list([x[0] for x in mc_inst.MC_sample[burnin:len(mc_inst.MC_sample):thinning]])
    diag_inst_lambda.set_variable_names(["lambda1","lambda2","lambda3"])
    diag_inst_lambda.show_traceplot((1,3))
    diag_inst_lambda.show_hist((1,3))
    diag_inst_lambda.print_summaries(3)

    diag_inst_Q = MCMC_Diag()
    diag_inst_Q.set_mc_samples_from_list([x[2].flatten() for x in mc_inst.MC_sample[burnin:len(mc_inst.MC_sample):thinning]])
    diag_inst_Q.set_variable_names(["q11","q12","q13","q21","q22","q23","q31","q32","q33"])
    diag_inst_Q.show_traceplot((3,3))
    diag_inst_Q.show_hist((3,3))
    diag_inst_Q.print_summaries(3)

    mc_z_samples = np.array([x[1] for x in mc_inst.MC_sample[burnin:len(mc_inst.MC_sample):thinning]])
    mc_z_prob = np.mean(mc_z_samples==0, axis=0)
    fig, ax = plt.subplots(4,1, figsize=(6,8))
    ax[0].plot([i+1 for i in range(len(fetal_lamb_movements))], fetal_lamb_movements)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("count")
    ax[0].set_title("Data")
    ax[1].plot([i+1 for i in range(len(fetal_lamb_movements))], np.mean(mc_z_samples==0, axis=0))
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("probability")
    ax[1].set_title("P[z_t=1|...]")
    ax[1].set_ylim(0,1)
    ax[2].plot([i+1 for i in range(len(fetal_lamb_movements))], np.mean(mc_z_samples==1, axis=0))
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("probability")
    ax[2].set_title("P[z_t=2|...]")
    ax[2].set_ylim(0,1)
    ax[3].plot([i+1 for i in range(len(fetal_lamb_movements))], np.mean(mc_z_samples==2, axis=0))
    ax[3].set_xlabel("time")
    ax[3].set_ylabel("probability")
    ax[3].set_title("P[z_t=3|...]")
    ax[3].set_ylim(0,1)
    plt.tight_layout()
    plt.show()

    def stationary(Q):
        k = Q.shape[0]
        P = Q - np.identity(k)
        P[:,-1] = np.ones(k)
        P_inv = np.linalg.inv(P)
        return P_inv[-1, ]
    stationary_prob_samples = [stationary(x[2]) for x in mc_inst.MC_sample[burnin:len(mc_inst.MC_sample):thinning]]
    diag_inst_st = MCMC_Diag()
    diag_inst_st.set_mc_samples_from_list(stationary_prob_samples)
    diag_inst_st.set_variable_names(["pi1", "pi2", "pi3"])
    diag_inst_st.show_hist((1,3))
    diag_inst_st.print_summaries(3)

    def marginal_prob_of_y(y, stprob_vec, lambda_vec):
        prob_sample = 0
        for pi, lam in zip(stprob_vec, lambda_vec):
            prob_sample += (pi*sp_pois.pmf(y, lam))
        return prob_sample
    prob_sample_at_y = []
    for y in [0,1,2,3,4,5,6,7,8]:
        prob_sample_at_y.append([])
        for stprob, lamb in zip(stationary_prob_samples, [x[0] for x in mc_inst.MC_sample[burnin:len(mc_inst.MC_sample):thinning]]):
            prob_sample_at_y[-1].append(marginal_prob_of_y(y, stprob, lamb))
    
    fig, ax = plt.subplots(3,1, figsize=(6,6))
    ax[0].plot([i+1 for i in range(len(fetal_lamb_movements))], fetal_lamb_movements)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("count")
    ax[0].set_title("Data")
    ax[1].plot([0,1,2,3,4,5,6,7,8], [np.sum(np.array(fetal_lamb_movements)==x) for x in [0,1,2,3,4,5,6,7,8]], 'o')
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("count")
    ax[1].set_ylim(0,len(fetal_lamb_movements))
    ax[1].set_title("Data")

    ax[2].plot([0,1,2,3,4,5,6,7,8], [np.quantile(x, (0.025, 0.975)) for x in prob_sample_at_y], 'o', color="gray")
    ax[2].plot([0,1,2,3,4,5,6,7,8], [np.mean(x) for x in prob_sample_at_y], 'o', color="red")
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("posterior marginal probability")
    ax[2].set_title("P[Y_t=y|...]")
    ax[2].set_ylim(0,1)
    plt.tight_layout()
    plt.show()
