import numpy as np
from scipy.stats import multivariate_normal as sp_mvn

class gaussian_process_simulator:
    def __init__(self, tau_cov_scale, phi_cov_range, alpha_cov_power):
        self.hyper_tau = tau_cov_scale
        self.hyper_phi = phi_cov_range
        self.hyper_alpha = alpha_cov_power
        self.grid = np.array([x/10 for x in range(1, 1000)])

    def mean_function1(self, t):
        return 0
    def mean_function2(self, t):
        return np.cos(t/10)*2
    def cov_function_power_exp(self, t1, t2):
        return self.hyper_tau**2 * np.exp(-self.hyper_phi * np.abs(t1-t2)**self.hyper_alpha)

    def run(self, n_num_path, mean_func = 1):
        if mean_func == 1:
            mean_function = self.mean_function1
        elif mean_func == 2:
            mean_function = self.mean_function2
        else:
            raise ValueError("select a correct mean function.")
        mean_vec = np.array([mean_function(t) for t in self.grid])
        cov_mat = np.array([[self.cov_function_power_exp(t1, t2) for t1 in self.grid] for t2 in self.grid])
        path = sp_mvn.rvs(mean = mean_vec, cov=cov_mat, size=n_num_path)
        return path
        


if __name__=="__main__":
    import matplotlib.pyplot as plt
    np.random.seed(20250209)
    select_mean_function = 2
    alpha2 = True
    alpha1 = False
    alpha15 = False
    alpha05 = False
    alpha_compare = False

    # alpha = 2 : exponential
    if alpha2:
        inst_1_1_2 = gaussian_process_simulator(1, 1, 2)
        x_1_1_2 = inst_1_1_2.run(3, select_mean_function)

        inst_10_1_2 = gaussian_process_simulator(10, 1, 2)
        x_10_1_2 = inst_10_1_2.run(3, select_mean_function)

        inst_1_01_2 = gaussian_process_simulator(1, 0.1, 2)
        x_1_01_2 = inst_1_01_2.run(3, select_mean_function)

        plt.plot(x_1_1_2.T, color="blue")
        plt.plot(x_10_1_2.T, color="orange")
        plt.show()

        plt.plot(x_1_1_2.T, color="blue")
        plt.plot(x_1_01_2.T, color="red")
        plt.show()
    
    # alpha = 1 : gaussian
    if alpha1:
        inst_1_1_1 = gaussian_process_simulator(1, 1, 1)
        x_1_1_1 = inst_1_1_1.run(3, select_mean_function)

        inst_10_1_1 = gaussian_process_simulator(10, 1, 1)
        x_10_1_1 = inst_10_1_1.run(3, select_mean_function)

        inst_1_10_1 = gaussian_process_simulator(1, 0.1, 1)
        x_1_10_1 = inst_1_10_1.run(3, select_mean_function)

        plt.plot(x_1_1_1.T, color="blue")
        plt.plot(x_10_1_1.T, color="orange")
        plt.show()

        plt.plot(x_1_1_1.T, color="blue")
        plt.plot(x_1_10_1.T, color="red")
        plt.show()
    
    # alpha = 1.5
    if alpha15:
        inst_1_1_15 = gaussian_process_simulator(1, 1, 1.5)
        x_1_1_15 = inst_1_1_15.run(3, select_mean_function)

        inst_10_1_15 = gaussian_process_simulator(10, 1, 1.5)
        x_10_1_15 = inst_10_1_15.run(3, select_mean_function)

        inst_1_10_15 = gaussian_process_simulator(1, 0.1, 1.5)
        x_1_10_15 = inst_1_10_15.run(3, select_mean_function)

        plt.plot(x_1_1_15.T, color="blue")
        plt.plot(x_10_1_15.T, color="orange")
        plt.show()

        plt.plot(x_1_1_15.T, color="blue")
        plt.plot(x_1_10_15.T, color="red")
        plt.show()
    
    # alpha = 0.5
    if alpha05:
        inst_1_1_05 = gaussian_process_simulator(1, 1, 0.5)
        x_1_1_05 = inst_1_1_05.run(3, select_mean_function)

        inst_10_1_05 = gaussian_process_simulator(10, 1, 0.5)
        x_10_1_05 = inst_10_1_05.run(3, select_mean_function)

        inst_1_10_05 = gaussian_process_simulator(1, 0.1, 0.5)
        x_1_10_05 = inst_1_10_05.run(3, select_mean_function)

        plt.plot(x_1_1_05.T, color="blue")
        plt.plot(x_10_1_05.T, color="orange")
        plt.show()

        plt.plot(x_1_1_05.T, color="blue")
        plt.plot(x_1_10_05.T, color="red")
        plt.show()
    
    #compare alpha
    if alpha_compare:
        inst_1_1_2 = gaussian_process_simulator(1, 1, 2)
        x_1_1_2 = inst_1_1_2.run(1, select_mean_function)

        inst_1_1_15 = gaussian_process_simulator(1, 1, 1.5)
        x_10_1_15 = inst_1_1_15.run(1, select_mean_function)

        inst_1_1_1 = gaussian_process_simulator(1, 1, 1)
        x_10_1_1 = inst_1_1_1.run(1, select_mean_function)

        inst_1_1_05 = gaussian_process_simulator(1, 1, 0.5)
        x_10_1_05 = inst_1_1_05.run(1, select_mean_function)


        plt.plot(x_1_1_2.T, color="blue", alpha=0.8)
        plt.plot(x_10_1_15.T, color="orange", alpha=0.8)
        plt.plot(x_10_1_1.T, color="red", alpha=0.8)
        plt.plot(x_10_1_05.T, color="green", alpha=0.8)
        plt.show()
