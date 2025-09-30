# %%
# %%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# %%
# Use LaTeX for plot labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# #############################################################################
# Model Parameters (from draw_phase2.py)
# #############################################################################
b = 0.45
h = 0.04
delta = 0.3
beta = 0.97829942**30
gamma = 0.33
eta = (1 + gamma + beta) / (1 + beta)
alpha = 0.33
A = 8 # Productivity is held constant for this analysis
z = 0.08
xi = h / delta
Q = xi * z * (1 + gamma + beta) / gamma

# #############################################################################
# Sigma Threshold Calculation (from draw_phase2.py)
# #############################################################################
def find_sigma_thres(SIGMA):
    # Ensure the term inside the square root is non-negative
    if (1 - SIGMA)**2 < 4 * Q:
        return np.nan
    X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * Q))
    expo = eta / alpha - 1 - alpha / (1 - alpha)
    return 1 - (((1 - alpha) / b)**(1 / alpha) * eta * ((A * beta * z * (1 - alpha) * (eta**alpha) / gamma)**(1 / (1 - alpha))) * ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

def solve_sigma_thres():
    guess = 0.1
    result, _, ier, _ = fsolve(lambda SIGMA: find_sigma_thres(SIGMA), guess, full_output=True)
    return result[0] if ier == 1 else np.nan

sigma1_thres = 1 - np.sqrt(4 * Q)
sigma2_thres = solve_sigma_thres()
sigma_hat = min(sigma1_thres, sigma2_thres)

# Set a base sigma for the first two plots
sigma_base = sigma_hat - 0.1

# #############################################################################
# Model Formulation (parameterized by sigma)
# #############################################################################
def kt1_dot(xt, kt, sigma_val):
    """Calculates the change in k, i.e., k_{t+1} - k_t."""
    denominator = gamma * ((1 - xt - sigma_val)**alpha)
    if denominator <= 0:
        return np.nan
    return A * beta * z * (1 - alpha) * (eta**alpha) * (kt**alpha) / denominator - kt

# #############################################################################
# Functions to find nullclines and steady states (parameterized by sigma)
# #############################################################################
def find_k_nullcline(x, sigma_val):
    """Given x and sigma, find k for the k-nullcline."""
    denominator = gamma * ((1 - x - sigma_val)**alpha)
    if denominator <= 0:
        return np.nan
    term = A * beta * z * (1 - alpha) * (eta**alpha) / denominator
    if term < 0:
        return np.nan
    return term**(1 / (1 - alpha))

def find_x_nullclines(sigma_val):
    """Finds the x values for the vertical x-nullclines for a given sigma."""
    discriminant = (1 - sigma_val)**2 - 4 * Q
    if discriminant < 0:
        return np.nan, np.nan
    delta_sqrt = np.sqrt(discriminant)
    x1 = (1 - sigma_val - delta_sqrt) / 2
    x2 = (1 - sigma_val + delta_sqrt) / 2
    return x1, x2

# #############################################################################
# --- PLOT 1: Construction of k-nullcline ---
# #############################################################################
fig1, ax1 = plt.subplots(figsize=(6, 4))
xt_vals_plot = np.linspace(0.01, 1 - sigma_base - 0.01, 100)
k_nullcline = [find_k_nullcline(x, sigma_base) for x in xt_vals_plot]

ax1.plot(xt_vals_plot, k_nullcline, color='black', lw=2)
ax1.set_xlabel(r'$x_t$', fontsize=18, loc='right')
ax1.set_ylabel(r'$k_t$', fontsize=18, rotation=0, loc='top')

# End points and limits
xmin, xmax = 0, 1 - sigma_base
ymin, ymax = 0, 2.5
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

# Annotations for k-dynamics
ax1.annotate(r'$k_{t+1} > k_t$', xy=(0.2, 0.3), fontsize=14)
ax1.annotate('', xy=(0.25, 0.5), xytext=(0.25, 0.8), arrowprops=dict(facecolor='black', arrowstyle='<-', lw=1.5))
ax1.annotate(r'$k_{t+1} < k_t$', xy=(0.2, 1.8), fontsize=14)
ax1.annotate('', xy=(0.25, 1.6), xytext=(0.25, 1.3), arrowprops=dict(facecolor='black', arrowstyle='<-', lw=1.5))
ax1.annotate(r'$k_{t+1} = k_t$', xy=(0.4, find_k_nullcline(0.35, sigma_base) + 0.2), fontsize=15, rotation=25)

ax1.set_yticks([])
ax1.set_xticks([])
ax1.spines[['right', 'top']].set_visible(False)

# add arrows to the tips of the axes
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig('fig_kPhi_full.pdf', bbox_inches='tight', dpi=300)


# %%
# #############################################################################
# --- PLOT 2: Construction of x-nullcline ---
# #############################################################################
fig2, ax2 = plt.subplots(figsize=(6, 4))
x1_ss, x2_ss = find_x_nullclines(sigma_base)

ax2.axvline(x=x1_ss, color='black', lw=2)
ax2.axvline(x=x2_ss, color='black', lw=2)
ax2.set_xlabel(r'$x_t$', fontsize=18, loc='right')
ax2.set_ylabel(r'$k_t$', fontsize=18, rotation=0, loc='top')

# End points and limits
xmin, xmax = 0, 1 - sigma_base
ymin, ymax = 0, 2.5
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

# Annotations for x-dynamics
ax2.annotate(r'$x_{t+1} > x_t$', xy=(x1_ss/2-0.03, 1.0), fontsize=14, rotation=90)
ax2.annotate('', xy=(x1_ss/2 + 0.02, 0.85), xytext=(x1_ss/2 - 0.02, 0.85), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
ax2.annotate(r'$x_{t+1} < x_t$', xy=((x1_ss+x2_ss)/2 - 0.03, 1.0), fontsize=14)
ax2.annotate('', xy=((x1_ss+x2_ss)/2 - 0.04, 0.85), xytext=((x1_ss+x2_ss)/2 + 0.04, 0.85), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))
ax2.annotate(r'$x_{t+1} > x_t$', xy=(x2_ss + 0.02, 1.0), fontsize=14, rotation=90)
ax2.annotate('', xy=(x2_ss + 0.05, 0.85), xytext=(x2_ss + 0.01, 0.85), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))

ax2.annotate(r'$x_{t+1}=x_t$', xy=(x1_ss, ymax), ha='center', fontsize=15)
ax2.annotate(r'$x_{t+1}=x_t$', xy=(x2_ss, ymax), ha='center', fontsize=15)

ax2.set_yticks([])
ax2.set_xticks([x1_ss, x2_ss])
ax2.set_xticklabels([r'$x_1^*$', r'$x_2^*$'], fontsize=14)
ax2.spines[['right', 'top']].set_visible(False)

# add arrows to the tips of the axes
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig('fig_kPsi_full.pdf', bbox_inches='tight', dpi=300)

# %%
# #############################################################################
# --- PLOT 3: Phase Plane Shift with Change in Sigma ---
# #############################################################################
sigma_old = sigma_hat - 0.15
sigma_new = sigma_hat - 0.01 # sigma_new > sigma_old

# Calculate nullclines and steady states for sigma_old
x1_old, x2_old = find_x_nullclines(sigma_old)
xt_vals_old = np.linspace(0.01, 1 - sigma_old - 0.01, 100)
k_nullcline_old = [find_k_nullcline(x, sigma_old) for x in xt_vals_old]
k1_old = find_k_nullcline(x1_old, sigma_old)
k2_old = find_k_nullcline(x2_old, sigma_old)

# Calculate nullclines and steady states for sigma_new
x1_new, x2_new = find_x_nullclines(sigma_new)
xt_vals_new = np.linspace(0.01, 1 - sigma_new - 0.01, 100)
k_nullcline_new = [find_k_nullcline(x, sigma_new) for x in xt_vals_new]
k1_new = find_k_nullcline(x1_new, sigma_new)
k2_new = find_k_nullcline(x2_new, sigma_new)


# Plot the phase portrait shift
fig3, ax3 = plt.subplots(figsize=(6, 4))


# Plot nullclines for sigma_old (solid black)
ax3.plot(xt_vals_old, k_nullcline_old, color='black', lw=2, label=r'$\sigma_1$')
ax3.axvline(x=x1_old, color='black', lw=2)
ax3.axvline(x=x2_old, color='black', lw=2)

# Plot nullclines for sigma_new (dashed blue)
ax3.plot(xt_vals_new, k_nullcline_new, color='blue', lw=2, linestyle='--', label=r'$\sigma_2 \in (\sigma_1,\hat{\sigma})$')
ax3.axvline(x=x1_new, color='blue', linestyle='--', lw=2)
ax3.axvline(x=x2_new, color='blue', linestyle='--', lw=2)
ax3.hlines(y=k1_old, xmin=0, xmax=x1_old, ls=':')
ax3.hlines(y=k1_new, xmin=0, xmax=x1_new, ls=':')

# Plot steady state points
ax3.scatter([x1_old, x2_old], [k1_old, k2_old], color='red', s=20, zorder=5)
ax3.scatter([x1_new, x2_new], [k1_new, k2_new], color='green', s=20, zorder=5)


# Set plot labels and limits
ax3.set_xlabel(r'$x_t$', fontsize=20, labelpad=-10, loc='right')
ax3.set_ylabel(r'$k_t$', fontsize=20, labelpad=-10, loc='top', rotation=0)
ax3.set_xlim(0, xt_vals_old[-1])
ax3.set_ylim(0, k2_old * 1.1)
ax3.spines[['right', 'top']].set_visible(False)

# Set ticks to show the steady state values
ax3.set_xticks([x1_old, x1_new])
ax3.set_xticklabels([r'$x_{1}^*$', r'$x_{3}^*$'], fontsize=12, rotation=0)
ax3.set_yticks([k1_old-0.02, k1_new+0.02])  # slightly change the values to better view k1 and k3
ax3.set_yticklabels([r'$k_{1}^*$', r'$k_{3}^*$'], fontsize=12)
ax3.tick_params(axis='y', pad=5)

## Annotate old and new ss
# annotate the steady state
ax3.annotate(r'old steady state', xy=(x1_old+0.009, k1_old), xytext=(x1_old+0.05, k1_old-0.5), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))
ax3.annotate(r'new steady state', xy=(x1_new+0.009, k1_new), xytext=(x1_new+0.05, k1_new-0.3), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))
ax3.annotate(r'unstable', xy=(x2_new, k2_new), xytext=(x2_new+0.05, k2_new), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))
ax3.annotate('', xy=(x2_old, k2_old), xytext=(x2_new+0.1, k2_new+0.1), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))
# set arrows that indicate the shifts of nullclines
ax3.annotate('', xy=(x1_old, 1.5), xytext=(x1_new, 1.5), arrowprops=dict(facecolor='black', arrowstyle='<-'))
ax3.annotate('', xy=(x1_old+0.3, k1_old+0.2), xytext=(x1_old+0.3, k1_new+0.3), arrowprops=dict(facecolor='black', arrowstyle='<-'))


# Add a legend
ax3.legend(fontsize=12, loc='upper center')

# End points and limits
xmin, xmax = 0, 1 - sigma_base
ymin, ymax = 0, 2.5
# add arrows to the tips of the axes
ax3.set_xlim(xmin, xmax)
ax3.set_ylim(ymin, ymax)
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))


plt.tight_layout()
plt.savefig('fig_kPhi_kPsi_sigma_changes_full.pdf', bbox_inches='tight', dpi=300)