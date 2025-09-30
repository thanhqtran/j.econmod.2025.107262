import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pathlib
import os

# ==========================================
# ======= Set current folder as root =======
# ==========================================
# Get the directory of the current script
script_dir = pathlib.Path(__file__).resolve().parent
os.chdir(script_dir)

# LaTeX for matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Parameters
b = 0.45
h = 0.04
delta = 0.3
beta = 0.97829942**30
gamma = 0.33
eta = (1 + gamma + beta) / (1 + beta)
alpha = 0.33
A = 8
z = 0.08

# Since we are going to run the model with different values of sigma,
# it is convenient to write a class to encapsulate the model's behavior.
# the Model class will take sigma as an input parameter and compute the necessary values.
# while keeping other parameters unchanged.

class Model:
    def __init__(self, sigma):
        self.sigma = sigma
        self.Q = (h / delta) * z * (1 + gamma + beta) / gamma
        self.sigma_hat = 1 - np.sqrt(4 * self.Q)
        self.check_param = 1 + eta / alpha - 2 / self.sigma_hat
        self.SIGMA_thres = self.solve_sigma_thres()

    def find_sigma_thres(self, SIGMA):
        X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * self.Q))
        expo = eta / alpha - 1 - alpha / (1 - alpha)
        return 1 - (((1 - alpha) / b)**(1 / alpha) * eta *
                    ((A * beta * z * (1 - alpha) * eta**alpha / gamma)**(1 / (1 - alpha))) *
                    ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

    def solve_sigma_thres(self):
        return fsolve(lambda SIGMA: self.find_sigma_thres(SIGMA), 0.2)[0]

    def x_t(self, n0):
        return h / (delta * n0)

    def k_hat_t(self, x):
        return ((b / (1 - alpha))**(1 / alpha)) * (1 / eta) * (((1 - x)**(eta / alpha)) /
                ((1 - x - self.sigma)**(eta / alpha - 1)))

    def n_t(self, x, k, phi, k_hat):
        if k <= k_hat:
            return (gamma / (z * (1 + gamma + beta))) * (1 - x - self.sigma * phi)
        else:
            return (gamma / (z * (1 + gamma + beta))) * (1 - x - self.sigma)

    def phi_t(self, x, k, k_hat):
        if k <= k_hat:
            return (((1 - alpha) / b)**(1 / alpha)) * (((1 - x - self.sigma)**(eta / alpha - 1)) /
                    (1 - x)**(eta / alpha)) * eta * k
        else:
            return 1

    def k_t1(self, x, k, phi, k_hat):
        if k <= k_hat:
            num = (1 - alpha) * ((1 - x - self.sigma)**(1 - alpha)) * eta**alpha * phi**(1 - alpha) * k**alpha
            denom = 1 - x - self.sigma * phi
            return (A * beta * z / gamma) * (num + (1 - x) * (1 - phi) * b) / denom
        else:
            return (A * beta * z / gamma) * ((1 - alpha) * eta**alpha * k**alpha) / ((1 - x - self.sigma)**alpha)

    def simulation(self):
        n0 = 2.1
        k0 = 0.1

        T = 2000
        nt = np.zeros(T)
        kt = np.zeros(T)
        phit = np.zeros(T)
        k_hatt = np.zeros(T)
        xt = np.zeros(T)
        time = np.arange(T)
        wage_gap = np.zeros(T)

        nt[0] = n0
        kt[0] = k0
        xt[0] = 0
        wage_gap[0] = 1

        for t in range(1, T):
            xt[t] = self.x_t(nt[t - 1])
            wage_gap[t] = ((1 - xt[t]) / (1 - xt[t] - self.sigma))**eta
            k_hatt[t] = self.k_hat_t(xt[t])
            phit[t] = self.phi_t(xt[t], kt[t - 1], k_hatt[t])
            nt[t] = self.n_t(xt[t], kt[t - 1], phit[t], k_hatt[t])
            kt[t] = self.k_t1(xt[t], kt[t - 1], phit[t], k_hatt[t])
        return kt, nt, xt, wage_gap, time, phit

    def cal_k1(self, n0, numvals=500):
        k0 = np.linspace(0.1, 2, numvals)
        k1 = np.zeros(numvals)
        for i in range(numvals):
            k = k0[i]
            x = self.x_t(n0)
            k_hat = self.k_hat_t(x)
            phi = self.phi_t(x, k, k_hat)
            k1[i] = self.k_t1(x, k, phi, k_hat)
        return k0, k1

    def plot_k(self, n0, numvals=500):
        k0 = np.linspace(0.1, 2, numvals)
        k1 = np.zeros(numvals)
        for i in range(numvals):
            k = k0[i]
            x = self.x_t(n0)
            k_hat = self.k_hat_t(x)
            phi = self.phi_t(x, k, k_hat)
            k1[i] = self.k_t1(x, k, phi, k_hat)

        fig, ax = plt.subplots()
        plt.plot(k0, k0, label='45-degree line', ls='--', color='black')
        plt.plot(k0, k1, label=r'$\Phi(k_t,n)$', color='blue')
        ax.spines[['right', 'top']].set_visible(False)
        ax.margins(x=0)
        plt.xticks([])
        plt.yticks([])
        ax.set_xlabel('$k_t$', fontsize=12, loc='right')
        ax.set_ylabel('$k_{t+1}$', fontsize=12, loc='top', rotation=0)
        plt.legend(fontsize=12)
        plt.show()

# === Simulation and Plotting ===
# initiate a list to store sigma and steady-state phi values

sigma_max = 0.51
sigma_test = []
phi_test = []
sigma_list = np.linspace(alpha / eta, sigma_max, 100)

# Run a sample simulation to extract the critical point
model0 = Model(sigma=0.3)
kt, nt, xt, wage_gap, time, phit = model0.simulation()
phi_hat = phit[-1]
sigma_hat1 = model0.sigma_hat
sigma_hat2 = model0.SIGMA_thres
# this is the sigma_hat from Proposition 3 in the paper
sigma_hat = min(sigma_hat1, sigma_hat2)

# Insert the critical point into the sigma_list
# Essentially, we are splitting the sigma_list into two parts:
# one part is less than sigma_hat and the other part is greater than sigma_hat
sigma_list = np.concatenate((
    sigma_list[sigma_list < sigma_hat],
    [sigma_hat],
    sigma_list[sigma_list > sigma_hat]
))

# Run the model for each sigma in sigma_list
# in each run, first save the sigma value,
# then run the simulation and save the last value of phi_t
for sigma in sigma_list:
    model = Model(sigma=sigma)
    _, _, _, _, _, phit = model.simulation()
    sigma_test.append(sigma)
    phi_test.append(phit[-1])   # only the last value of phi_t is needed

# Plotting
fig = plt.figure(figsize=(4, 3))
plt.plot(sigma_test, phi_test, color='black')
plt.scatter(sigma_hat, phi_hat, color='red', label='critical point', s=15)
plt.xlabel(r'$\sigma$', fontsize=13, loc='right', rotation=0, labelpad=-15)
plt.ylabel(r'$\phi^{\ast}$', fontsize=13, loc='top', rotation=0, labelpad=-15)
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)
ax.margins(x=0)
plt.tight_layout()
ax.vlines(sigma_hat, 0.25, phi_hat, linestyle='--', color='grey')
plt.xticks([alpha / eta, sigma_hat], [r'$\frac{\alpha}{\eta}$', r'$\hat{\sigma}$'], fontsize=12)
plt.yticks([phi_hat], [1], fontsize=12)
plt.scatter(sigma_hat, 0.25, color='red', s=15)
# aethetic adjustments
xmin = alpha / eta
xmax = sigma_max+0.02
ymin = 0.25
ymax = 1.1
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
# Add arrows to the tips of the axes
ax.annotate('', xy=(xmax, ymin), xytext=(xmax - 0.01, ymin),
            arrowprops=dict(arrowstyle='->', lw=1))
ax.annotate('', xy=(xmin, ymax), xytext=(xmin, ymax - 0.01),
            arrowprops=dict(arrowstyle='->', lw=1))
plt.savefig('fig_phi_sigma_eqm_test.pdf', format='pdf', bbox_inches='tight', dpi=300)