import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# LaTeX formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# === Parameters ===
b = 0.45
h = 0.04
delta = 0.3
beta = 0.97829942**30
gamma = 0.33
eta = (1 + gamma + beta) / (1 + beta)
alpha = 0.33
A = 8
z = 0.08
Q = h / delta * z * (1 + gamma + beta) / gamma

# key params
sigma = 1 - (8 / 4.5)**(-1)

def find_sigma_thres(SIGMA):
    X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * Q))
    expo = eta / alpha - 1 - alpha / (1 - alpha)
    return 1 - (((1 - alpha) / b)**(1 / alpha) * eta * ((A * beta * z * (1 - alpha) * (eta**alpha) / gamma)**(1 / (1 - alpha))) * ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

def solve_sigma_thres():
    return fsolve(lambda SIGMA: find_sigma_thres(SIGMA), sigma)[0]

sigma1 = 1 - np.sqrt(4 * Q)
sigma2 = solve_sigma_thres()
sigma_thres = min(sigma1, sigma2)

# === Common Functions ===
def x_t(n):
    return h / (delta * n)

def k_hat_t(x, sigma):
    return ((b / (1 - alpha))**(1 / alpha)) / eta * ((1 - x)**(eta / alpha)) / ((1 - x - sigma)**(eta / alpha - 1))

def phi_t(x, k, k_hat, sigma):
    if k < k_hat:
        return ((1 - alpha) / b)**(1 / alpha) * ((1 - x - sigma)**(eta / alpha - 1)) / ((1 - x)**(eta / alpha)) * eta * k
    else:
        return 1

def n_t(x, k, phi, k_hat, sigma):
    if k < k_hat:
        return gamma / (z * (1 + gamma + beta)) * (1 - x - sigma * phi)
    else:
        return gamma / (z * (1 + gamma + beta)) * (1 - x - sigma)

def k_t1(x, k, phi, k_hat, sigma):
    coeff = A * beta * z / gamma
    if k < k_hat:
        numerator = (1 - alpha) * (1 - x - sigma)**(1 - alpha) * eta**alpha * phi**(1 - alpha) * k**alpha
        numerator += (1 - x) * (1 - phi) * b
        denominator = 1 - x - sigma * phi
        return coeff * numerator / denominator
    else:
        return coeff * (1 - alpha) * eta**alpha * k**alpha / ((1 - x - sigma)**alpha)

# === Simulation Function ===
def simulate_dynamics(sigma, T=8, n0=2.2, k0=0.1):
    n = np.zeros(T)
    k = np.zeros(T)
    phi = np.zeros(T)
    k_hat = np.zeros(T)
    x = np.zeros(T)
    wage_gap = np.zeros(T)
    time = np.arange(T)

    n[0] = n0
    k[0] = k0
    x[0] = 0
    wage_gap[0] = 1

    for t in range(1, T):
        x[t] = x_t(n[t - 1])
        k_hat[t] = k_hat_t(x[t], sigma)
        phi[t] = phi_t(x[t], k[t - 1], k_hat[t], sigma)
        n[t] = n_t(x[t], k[t - 1], phi[t], k_hat[t], sigma)
        k[t] = k_t1(x[t], k[t - 1], phi[t], k_hat[t], sigma)
        wage_gap[t] = ((1 - x[t]) / (1 - x[t] - sigma))**eta

    return time, n, phi, x, k, wage_gap

# === Simulate for Two Sigma Values ===
T = 8
time_low, n_low, phi_low, x_low, k_low, _ = simulate_dynamics(sigma=0.3, T=T)
time_high, n_high, phi_high, x_high, k_high, _ = simulate_dynamics(sigma=0.4, T=T)

# === Plotting ===
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

axs[0, 0].plot(time_low, n_low, 'k-', label=r'low $\sigma$')
axs[0, 0].plot(time_high, n_high, 'k--', label=r'high $\sigma$')
axs[0, 0].set_title(r'Average fertility $n_t$')
axs[0, 0].set_xlabel(r'period')
axs[0, 0].legend()

axs[0, 1].plot(time_low, phi_low, 'k-')
axs[0, 1].plot(time_high, phi_high, 'k--')
axs[0, 1].set_xlabel(r'period')
axs[0, 1].set_title(r'Regular worker ratio $\phi_t$')

axs[1, 0].plot(time_low, x_low, 'k-')
axs[1, 0].plot(time_high, x_high, 'k--')
axs[1, 0].set_xlabel(r'period')
axs[1, 0].set_title(r'Average elderly care time cost $x_t$')

axs[1, 1].plot(time_low, k_low, 'k-')
axs[1, 1].plot(time_high, k_high, 'k--')
axs[1, 1].set_xlabel(r'period')
axs[1, 1].set_title(r'Capital $k_t$')

plt.tight_layout()
plt.savefig('fig_benchmark_simul_A_update.pdf', format='pdf', bbox_inches='tight', dpi=300)