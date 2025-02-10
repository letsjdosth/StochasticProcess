import numpy as np
from scipy.stats import norm as sp_norm
from scipy.stats import multivariate_normal as sp_mvn
from scipy.stats import invgamma as sp_invgam

import matplotlib.pyplot as plt
from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag


#generate data
np.random.seed(20250212)
n_sample_size = 100
x_data = sp_norm.rvs(loc=0, scale=1, size=n_sample_size)
y_data = 0.3+0.4*x_data+0.5*np.sin(2.7*x_data)+1.1/(1+x_data**2) + sp_norm.rvs(loc=0, scale=0.2, size=n_sample_size)


class gaussian_process_post(MCMC_Gibbs):
    """
    == standard setting ==
    t = x_i : 1-dim
    y_t : 1-dim
    y_t ~ N(theta_t, sigma^2), iid
    theta ~ GP(mean(t)=mu, cov(t1,t2)= power-exp(tau, phi, alpha)
    and
    sigma2 ~ inv.gam(a_sigma,b_sigma)
    mu ~ N(a_mu, b_mu)
    tau2 ~ inv.gam(a_tau, b_tau)
    phi ~ unif(0, b_phi)
    alpha is fixed as a hyperparameter
    """
    def __init__(self, initial, t_time_idx, y_data):
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        self.t_time_idx = t_time_idx #np array
        self.y_data = y_data #np array
        self.n = len(y_data)

        #hyperparamters
        self.hyper_a_sigma = 1
        self.hyper_b_sigma = 1
        self.hyper_a_mu = 0
        self.hyper_b_mu = 1
        self.hyper_a_tau = 1
        self.hyper_b_tau = 1
        self.hyper_b_phi = 6 #be careful!
        self.hyper_alpha = 2 #be careful!

        self.MC_sample = []
        initial = initial + [self.make_inv_H_corr_mat(initial[4])]
        self.MC_sample.append(initial)

    def corr_function_power_exp(self, t1, t2, phi):
        return np.exp(-phi * (np.abs(t1-t2)**self.hyper_alpha))

    def make_inv_H_corr_mat(self, phi):
        corr_mat = np.array([[self.corr_function_power_exp(t1, t2, phi) for t1 in self.t_time_idx] for t2 in self.t_time_idx])
        corr_mat = corr_mat + np.identity(self.n)*0.01
        L_inv = np.linalg.inv(np.linalg.cholesky(corr_mat))
        inv_mat = np.dot(L_inv.T, L_inv)
        return inv_mat

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        new[0] = np.array(new[0])
        new[5] = np.array(new[5])
        #update new
        new = self.full_conditional_sampler_theta(new)
        new = self.full_conditional_sampler_sigma2(new)
        new = self.full_conditional_sampler_mu(new)
        new = self.full_conditional_sampler_tau2(new)
        new = self.full_conditional_sampler_phi(new)
        self.MC_sample.append(new)

    def full_conditional_sampler_theta(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        #update new
        new_cov = np.linalg.inv(np.identity(self.n)/last_param[1] + last_param[5]/last_param[3])
        new_mean = new_cov @ (self.y_data/last_param[1] + last_param[2]*(last_param[5] @ np.ones((self.n, )))/last_param[3]) 
        new_theta = sp_mvn.rvs(mean = new_mean, cov = new_cov)
        new_sample[0] = new_theta
        return new_sample
    
    def full_conditional_sampler_sigma2(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        #update new
        new_a = self.hyper_a_sigma + 0.5*self.n
        new_b = self.hyper_b_sigma + 0.5*np.sum((self.y_data - last_param[0])**2)
        new_sigma2 = sp_invgam.rvs(a = new_a, scale = new_b) #check if it is new_b or 1/new_b (maybe new_b)
        new_sample[1] = new_sigma2
        return new_sample
    
    def full_conditional_sampler_mu(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        #update new
        btm = last_param[3] + self.hyper_b_mu * np.ones((1, self.n)) @ last_param[5] @ np.ones((self.n, 1))
        new_var = last_param[3] * self.hyper_b_mu / btm
        new_mean = (last_param[3] * self.hyper_a_mu + self.hyper_b_mu * last_param[0] @ last_param[5] @ np.ones((self.n, 1))) / btm
        new_mu = sp_norm.rvs(loc = new_mean, scale = new_var**0.5)
        new_sample[2] = new_mu
        return new_sample

    def full_conditional_sampler_tau2(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        #update new
        quad_one_term = np.reshape(last_param[0] - last_param[2], (1, self.n))
        new_a = self.hyper_a_tau + 0.5 * self.n
        new_b = self.hyper_b_tau + 0.5 * quad_one_term @ last_param[5] @ np.transpose(quad_one_term)
        new_tau2 = sp_invgam.rvs(a = new_a, scale = new_b) #check if it is new_b or 1/new_b (maybe new_b)
        new_sample[3] = new_tau2
        return new_sample
    
    def full_conditional_sampler_phi(self, last_param):
        new_sample = [x for x in last_param]
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        #update new
        new_phi = 0.5
        new_sample[4] = new_phi
        new_sample[5] = self.make_inv_H_corr_mat(new_phi)
        return new_sample
    

class post_gaussian_process_simulator:
    def __init__(self, post_sample, hyper_alpha, x_data):
        #param
        # 0      1       2   3     4--->5
        #[theta, sigma2, mu, tau2, phi, H-inv]
        self.post = post_sample
        self.grid = np.array([x/5 - 3 for x in range(0, 30)])
        self.m = len(self.grid)
        self.n = len(post_sample[0][0])
        
        self.x_data = x_data
        self.hyper_alpha = hyper_alpha

    def cov_function_power_exp(self, t1, t2, phi):
        return np.exp(-phi * np.abs(t1-t2)**self.hyper_alpha)

    def run(self, sample_idx):
        post_i = self.post[sample_idx]
        Hstar = np.array([[self.cov_function_power_exp(tn, tm, post_i[4]) for tn in self.x_data] for tm in self.grid])
        Htilde = np.array([[self.cov_function_power_exp(t1, t2, post_i[4]) for t1 in self.grid] for t2 in self.grid])
        # print(self.Hstar.shape) #m x n
        pred_mean = post_i[2] * np.ones(self.m) + Hstar @ post_i[5] @ (post_i[0] - post_i[2])

        # # Compute ABA^T
        # ABA_T = np.dot(A, np.dot(B, A.T))
        # # Symmetrize the resulting matrix
        # ABA_T_sym = (ABA_T + ABA_T.T) / 2
        # hhh = np.dot(Hstar, np.dot(post_i[5], Hstar.T))
        # hhh_sym = (hhh + hhh.T) / 2
        
        # return pred_mean
    
        hhh_sym = Hstar @ post_i[5] @ np.transpose(Hstar)
        pred_cov = post_i[3] * (Htilde - hhh_sym)
        path = sp_mvn.rvs(mean = pred_mean, cov = (pred_cov + 0.1*np.identity(self.m)))
        return path#, pred_mean
        


if __name__=="__main__":
    gibbs_initial_param = [np.zeros(n_sample_size), 1, 0, 1, 2] #no H_inv here
    inst = gaussian_process_post(gibbs_initial_param, x_data, y_data)
    inst.generate_samples(500)
    plt.scatter(x_data, y_data)
    plt.scatter(x_data, inst.MC_sample[499][0], color="red")
    plt.show()

    inst2 = post_gaussian_process_simulator(inst.MC_sample, hyper_alpha=2, x_data=x_data)
    plt.scatter(x_data, y_data)
    plt.plot(inst2.grid, inst2.run(419))
    plt.plot(inst2.grid, inst2.run(439))
    plt.plot(inst2.grid, inst2.run(459))
    plt.plot(inst2.grid, inst2.run(479))
    plt.plot(inst2.grid, inst2.run(499))
    plt.show()
