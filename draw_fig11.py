import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==========================================
# ======= Set current folder as root =======
# ==========================================
# Get the directory of the current script
script_dir = pathlib.Path(__file__).resolve().parent
os.chdir(script_dir)

# ==========================================
# ============ Load target data ============
# ========================================== 
# Ensure you have the 'calibration_target.xlsx' file in the same directory as this script
# Data for Japan's fertility rate: https://www.macrotrends.net/global-metrics/countries/jpn/japan/fertility-rate#:~:text=Japan%20fertility%20rate%20for%202025,a%200%25%20increase%20from%202021.
# Data for Japan's non-regular worker ratio: https://www.sciencedirect.com/science/article/abs/pii/S0889158320300253
raw_dat = pd.DataFrame(pd.read_excel('calibration_target.xlsx'))
# interpolate missing values in non-regular worker ratio
target_data = raw_dat.interpolate(method='linear', limit_direction='both')
target_data = target_data[target_data['year'] >= 1970].reset_index(drop=True)

# ==========================================
# ============ Prepare for plots ===========
# ========================================== 

# Since one period in the model is 30 years, we need to adjust the time_x accordingly
# create a linspace of 0, 1, 2 with 30 points between 0 and 1, 1 and 2
periods = 30
xspace_1 = np.linspace(0, 1, periods)
xspace_2 = np.linspace(1, 2, periods)
xspace_3 = np.linspace(2, 3, periods)
# combine
xspace = np.concatenate([xspace_1, xspace_2[1:len(target_data['fertility'])-periods+1]])

target_data['time_x'] = xspace
target_data['non_regular_pct'] = target_data['non_regular_rates'] / 100
target_data['regular_pct'] = 1 - target_data['non_regular_pct']

# ==========================================
# ============ Model's block ===============
# ==========================================
# PARAMETERS
b = 0.45
h = 0.04
delta = 0.3
beta = 0.97829942**30
gamma = 0.33
eta = (1 + gamma + beta) / (1 + beta)
alpha = 0.33
A = 8
z = 0.08
# subparameter
Q = h / delta * z * (1 + gamma + beta) / gamma
# key params
sigma = 1 - (8 / 4.5)**(-1)

# Some helper functions
# This block solves the model's threshold for sigma
def find_sigma_thres(SIGMA):
    X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * Q))
    expo = eta / alpha - 1 - alpha / (1 - alpha)
    return 1 - (((1 - alpha) / b)**(1 / alpha) * eta * ((A * beta * z * (1 - alpha) * (eta**alpha) / gamma)**(1 / (1 - alpha))) * ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

def solve_sigma_thres():
    return fsolve(lambda SIGMA: find_sigma_thres(SIGMA), sigma)[0]

sigma1 = 1 - np.sqrt(4 * Q)
sigma2 = solve_sigma_thres()
sigma_thres = min(sigma1, sigma2)   # corresponds to sigma hat in the paper

# Compute elderly care time (eq.(6))
def x_t(n0):
    return h / (delta * n0)

# Compute k_hat (eq.(17))
def k_hat_t(x):
    return ((b / (1 - alpha))**(1 / alpha)) * (1 / eta) * (((1 - x)**(eta / alpha)) / ((1 - x - sigma)**(eta / alpha - 1)))

# Compute average fertility (eq.(20))
def n_t(x, k, phi, k_hat):
    if k < k_hat:
        return (gamma / (z * (1 + gamma + beta))) * (1 - x - sigma * phi)
    else:
        return (gamma / (z * (1 + gamma + beta))) * (1 - x - sigma)

# Compute regular worker ratio (eq.(18))
def phi_t(x, k, k_hat):
    if k < k_hat:
        return (((1 - alpha) / b)**(1 / alpha)) * (((1 - x - sigma)**(eta / alpha - 1)) / (1 - x)**(eta / alpha)) * eta * k
    else:
        return 1

# Compute k(t+1) (eq.(24))
def k_t1(x, k, phi, k_hat):
    if k < k_hat:
        return (A * beta * z / gamma) * ((1 - alpha) * ((1 - x - sigma)**(1 - alpha)) * (eta**alpha) * (phi**(1 - alpha)) * (k**alpha) + (1 - x) * (1 - phi) * b) / (1 - x - sigma * phi)
    else:
        return (A * beta * z / gamma) * ((1 - alpha) * (eta**alpha) * (k**alpha)) / ((1 - x - sigma)**alpha)

# Capital per regular worker
def kphi_(k, phi):
    return k / phi

# a function to solve for initial k given phi and n
def find_inik(phi, n):
    x = x_t(n)
    theta_x = (1-x-sigma)**(eta/alpha-1) / ((1-x)**(eta/alpha))
    if phi < 1:
        return phi / (theta_x * ((1 - alpha) / b)**(1 / alpha) * eta) / eta
    else:
        return (b/(1-alpha))**(1/alpha) * (1/eta) * (1/theta_x)
    


# ==========================================
# ============ Model's dynamics ============
# ==========================================
# Initial conditions
n0 = target_data['fertility'][0]
k0 = find_inik(target_data['regular_pct'][0], n0)
print(n0, k0, target_data['regular_pct'][0])


# initiate arrays for storing results
T_MAX = 500
nt = np.zeros(T_MAX)
kt = np.zeros(T_MAX)
phit = np.zeros(T_MAX)
k_hatt = np.zeros(T_MAX)
xt = np.zeros(T_MAX)
time = np.arange(T_MAX)
wage_gap = np.zeros(T_MAX)
k_reg_t = np.zeros(T_MAX)

# Set initial conditions (at time zero)
nt[0] = n0
kt[0] = k0
xt[0] = x_t(2.142)  # n(-1)=2.142
time[0] = 0
wage_gap[0] = 2
phit[0] = 0.917
k_reg_t[0] = kphi_(kt[0], phit[0])

# Iterate over time to compute the model's dynamics
for t in np.arange(1, T_MAX):
    time[t] = t
    xt[t] = x_t(nt[t - 1])
    wage_gap[t] = ((1 - xt[t]) / (1 - xt[t] - sigma))**eta
    k_hatt[t] = k_hat_t(xt[t])
    phit[t] = phi_t(xt[t], kt[t - 1], k_hatt[t])
    nt[t] = n_t(xt[t], kt[t - 1], phit[t], k_hatt[t])
    kt[t] = k_t1(xt[t], kt[t - 1], phit[t], k_hatt[t])
    k_reg_t[t] = kphi_(kt[t], phit[t])

# ==========================================
# ============ PLOTTING ====================
# ==========================================

# LaTeX formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plot nt, xt, kt, phit in a 4 by 4 grid
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

T = 5   # number of periods to plot

axs[0, 0].plot(time[:T + 1], nt[:T + 1], color='black', label='model')
axs[0, 0].hlines(nt[T_MAX - 1], xmin=0, xmax=T, color='gray', ls=':')
axs[0, 0].annotate(round(nt[T_MAX - 1], 2), (T, nt[T_MAX - 1]), textcoords="offset points",
                   xytext=(0, 5), ha='right', fontsize=10, color='black')
for x, y in zip(target_data['time_x'], target_data['fertility']):
    axs[0, 0].scatter(x, y, color='gray', s=5)
axs[0, 0].scatter(target_data['time_x'], target_data['fertility'], color='blue', s=5, label='data')
axs[0, 0].set_xlabel('period')
axs[0, 0].legend()
axs[0, 0].set_title(r'Average fertility $n_t$')

axs[0, 1].plot(time[:T + 1], phit[:T + 1], color='black', label='model')
for x, y in zip(target_data['time_x'], target_data['regular_pct']):
    axs[0, 1].scatter(x, y, color='red', s=5)
axs[0, 1].scatter(target_data['time_x'], target_data['regular_pct'], color='red', s=5, label='data')
axs[0, 1].hlines(phit[T_MAX - 1], xmin=0, xmax=T, color='gray', ls=':')
axs[0, 1].set_xlabel('period')
axs[0, 1].annotate(round(phit[T_MAX - 1], 2), (T, phit[T_MAX - 1]), textcoords="offset points",
                   xytext=(0, 5), ha='right', fontsize=10, color='black')
axs[0, 1].legend()
axs[0, 1].set_title(r'Regular worker ratio $\phi_t$')

axs[1, 0].plot(time[:T + 1], xt[:T + 1], color='black')
axs[1, 0].set_title(r'Average elderly care time cost $x_t$')
axs[1, 0].hlines(xt[T_MAX - 1], xmin=0, xmax=T, color='gray', ls=':')
axs[1, 0].set_xlabel('period')
axs[1, 0].annotate(round(xt[T_MAX - 1], 2), (T, xt[T_MAX - 1]), textcoords="offset points",
                   xytext=(0, 5), ha='right', fontsize=10, color='black')
axs[1, 0].set_ylim(0, 0.2)

axs[1, 1].plot(time[:T + 1], k_reg_t[:T + 1], color='black', label='$k^r_t$', ls='--')
axs[1, 1].plot(time[:T + 1], kt[:T + 1], color='black', label='$k_t$')
axs[1, 1].set_xlabel('period')
axs[1, 1].set_title(r'Capital per worker ($k_t$ and $k^r_t$)')
axs[1, 1].legend(loc='upper left', fontsize=8, ncol=1, frameon=False)

plt.tight_layout()
plt.savefig('fig_benchmark_simul_japan_A.pdf', format='pdf', bbox_inches='tight', dpi=300)