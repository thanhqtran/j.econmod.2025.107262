# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# %%
# Use LaTeX for plot labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# #############################################################################
# Model Parameters (from draw_phase1.py)
# #############################################################################
b = 0.45
h = 0.04
delta = 0.3
beta = 0.97829942**30
gamma = 0.33
eta = (1 + gamma + beta) / (1 + beta)
alpha = 0.33
A = 8
z = 0.08
xi = h / delta
Q = xi * z * (1 + gamma + beta) / gamma

# #############################################################################
# Function to find the sigma threshold (from draw_phase1.py)
# #############################################################################
def find_sigma_thres(SIGMA):
    # Handle potential domain errors for sqrt
    if (1 - SIGMA)**2 - 4 * Q < 0:
        return np.nan
    X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * Q))
    expo = eta / alpha - 1 - alpha / (1 - alpha)
    # Handle potential domain errors for power
    if (1 - X - SIGMA) < 0:
        return np.nan
    return 1 - (((1 - alpha) / b)**(1 / alpha) * eta * ((A * beta * z * (1 - alpha) * (eta**alpha) / gamma)**(1 / (1 - alpha))) * ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

def solve_sigma_thres():
    guess = 0.1
    result, _, ier, _ = fsolve(lambda SIGMA: find_sigma_thres(SIGMA), guess, full_output=True)
    return result[0] if ier == 1 else np.nan

sigma1_thres = 1 - np.sqrt(4 * Q) if 4 * Q <= 1 else np.nan
sigma2_thres = solve_sigma_thres()
sigma_hat = np.nanmin([sigma1_thres, sigma2_thres])

# Use a single sigma for the first two plots
sigma = sigma_hat + 0.05

# #############################################################################
# Model Formulation (from draw_phase1.py, now parameterized by sigma)
# #############################################################################
def phi_t(xt, kt, sigma):
    """Calculates the phi_t term."""
    # Add checks to prevent division by zero or negative powers
    if 1 - xt <= 0 or 1 - xt - sigma <= 0:
        return np.nan
    return (((1 - alpha) / b) ** (1 / alpha)) * ((1 - xt - sigma) ** ((eta / alpha) - 1)) / ((1 - xt) ** (eta / alpha)) * eta * kt

def xt1_dot(xt, kt, sigma):
    """Calculates the change in x, i.e., x_{t+1} - x_t."""
    phi = phi_t(xt, kt, sigma)
    if np.isnan(phi) or (1 - xt - sigma * phi) == 0:
        return np.nan
    return Q / (1 - xt - sigma * phi) - xt

def kt1_dot(xt, kt, sigma):
    """Calculates the change in k, i.e., k_{t+1} - k_t."""
    phi = phi_t(xt, kt, sigma)
    denominator = 1 - xt - sigma * phi
    if np.isnan(phi) or denominator == 0:
        return np.nan
    return (A * beta * z * b / gamma) * ((1 - xt) * ((1 - xt - sigma)**(1 - eta)) * phi + (1 - xt) * (1 - phi)) / denominator - kt

# #############################################################################
# Functions to find nullclines and steady state (parameterized by sigma)
# #############################################################################
def find_kstar(x, sigma):
    """Given x and sigma, find k for the k-nullcline."""
    sol, _, ier, _ = fsolve(lambda k: kt1_dot(x, k, sigma), 0.6, full_output=True)
    return sol[0] if ier == 1 and sol[0] > 0 else np.nan

def find_xstar(k, sigma):
    """Given k and sigma, find x for the x-nullcline."""
    sol, _, ier, _ = fsolve(lambda x: xt1_dot(x, k, sigma), 0.1, full_output=True)
    return sol[0] if ier == 1 and sol[0] > 0 else np.nan

def find_steady_state(sigma):
    """Finds the steady state for a given sigma."""
    def system(vars):
        x, k = vars
        return [xt1_dot(x, k, sigma), kt1_dot(x, k, sigma)]
    initial_guess = [0.1, 0.5]
    solution, _, ier, _ = fsolve(system, initial_guess, full_output=True)
    return solution if ier == 1 else (np.nan, np.nan)

# #############################################################################
# --- PLOT 1: Construction of k-nullcline ---
# #############################################################################
fig1, ax1 = plt.subplots(figsize=(6,4))
xt_vals = np.linspace(0.01, 1 - sigma - 0.01, 100)
k_nullcline = [find_kstar(x, sigma) for x in xt_vals]

ax1.plot(xt_vals, k_nullcline, color='black', lw=2)
ax1.set_xlabel(r'$x_t$', fontsize=18, loc='right')
ax1.set_ylabel(r'$k_t$', fontsize=18, rotation=0, loc='top', labelpad=20)

# end points
xmin = 0
xmax = 1 - sigma
ymin = 0
ymax = 2

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

# Annotations for k-dynamics
ax1.annotate(r'$k_{t+1} > k_t$', xy=(0.15, 0.2), fontsize=14)
ax1.annotate('', xy=(0.2, 0.4), xytext=(0.2, 0.6), arrowprops=dict(facecolor='black', arrowstyle='<-'))
ax1.annotate(r'$k_{t+1} < k_t$', xy=(0.15, 1.05), fontsize=14)
ax1.annotate('', xy=(0.2, 1), xytext=(0.2, 0.8), arrowprops=dict(facecolor='black', arrowstyle='<-'))
# annotate at the tip of the curve, rotate the text
plt.annotate(r'$k_\Psi = \Phi(x_t, k_t)$', xy=(0.4, 0.55), xytext=(0.4, 0.55), fontsize=15, rotation=-15)
ax1.set_yticks([])
ax1.set_xticks([])

# add arrows to the tips of the axes
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))


ax1.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig('fig_kPhi_part.pdf', bbox_inches='tight', dpi=300)

# #############################################################################
# --- PLOT 2: Construction of x-nullcline ---
# #############################################################################
fig2, ax2 = plt.subplots(figsize=(6, 4))
kt_vals = np.linspace(0.01, 2, 100)
x_nullcline = [find_xstar(k, sigma) for k in kt_vals]

ax2.plot(x_nullcline, kt_vals, color='black', lw=2.5)
ax2.set_xlabel(r'$x_t$', fontsize=18, loc='right')
ax2.set_ylabel(r'$k_t$', fontsize=18, rotation=0, loc='top', labelpad=20)

# end points
xmin = 0
xmax = 1 - sigma
ymin = 0
ymax = 2

ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

# Annotations for x-dynamics
ax2.annotate(r'$x_{t+1} > x_t$', xy=(0.01, 1.0), fontsize=14)
ax2.annotate('', xy=(0.01, 0.85), xytext=(0.05, 0.85), arrowprops=dict(facecolor='black', arrowstyle='<-'))
ax2.annotate(r'$x_{t+1} < x_t$', xy=(0.2, 1.0), fontsize=14)
ax2.annotate('', xy=(0.25, 0.85), xytext=(0.3, 0.85), arrowprops=dict(facecolor='black', arrowstyle='->'))

# annotate at the tip of the curve, rotate the text
plt.annotate(r'$x_\Psi = \Psi(x_t, k_t)$', xy=(0.23, 1.4),xytext=(0.23, 1.4), fontsize=15, rotation=43)
ax2.set_yticks([])
ax2.set_xticks([])

# add arrows to the tips of the axes
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))

ax2.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig('fig_kPsi_part.pdf', bbox_inches='tight', dpi=300)

# #############################################################################
# --- PLOT 3: Phase Plane Shift with Change in Sigma ---
# #############################################################################
sigma_old = sigma_hat + 0.05
sigma_new = sigma_hat + 0.16

# Calculate steady states for both sigma values
x_ss_old, k_ss_old = find_steady_state(sigma_old)
x_ss_new, k_ss_new = find_steady_state(sigma_new)

# Create grid for plotting
T = 100
xt_vals_old = np.linspace(0.01, 1 - sigma_old - 0.01, T)
xt_vals_new = np.linspace(0.01, 1 - sigma_new - 0.01, T)
kt_vals_plot = np.linspace(0.01, 2, T)

# Calculate nullclines for both sigma values
k_nullcline_old = [find_kstar(x, sigma_old) for x in xt_vals_old]
x_nullcline_old = [find_xstar(k, sigma_old) for k in kt_vals_plot]
k_nullcline_new = [find_kstar(x, sigma_new) for x in xt_vals_new]
x_nullcline_new = [find_xstar(k, sigma_new) for k in kt_vals_plot]

# Plot the phase portrait shift
fig3, ax3 = plt.subplots(figsize=(6, 4))

# Plot nullclines
ax3.plot(xt_vals_old, k_nullcline_old, color='black', lw=2, label=r'$\sigma_1$')
ax3.plot(x_nullcline_old, kt_vals_plot, color='black', lw=2)
ax3.plot(xt_vals_new, k_nullcline_new, color='blue', lw=2, linestyle='--', label=r'$\sigma_2 > \sigma_1$')
ax3.plot(x_nullcline_new, kt_vals_plot, color='blue', lw=2, linestyle='--')

# Plot steady state points
ax3.scatter(x_ss_old, k_ss_old, color='red', s=20, zorder=5)
ax3.scatter(x_ss_new, k_ss_new, color='green', s=20, zorder=5)

# Add an arrow to show the shift in the steady state
#ax3.annotate('', xy=(x_ss_new, k_ss_new), xytext=(x_ss_old, k_ss_old), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1, ls='dotted'))

# end points
xmin = 0
xmax = xt_vals_new[-1]
ymin = 0
ymax = 2

# add arrows to the tips of the axes
plt.annotate('', xy=(xmin, ymin), xytext=(xmax, ymin), arrowprops=dict(arrowstyle="<-"))
plt.annotate('', xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->"))

# Set plot labels and limits from draw_phase1.py
ax3.set_xlabel(r'$x_t$', fontsize=20, labelpad=-10, loc='right')
ax3.set_ylabel(r'$k_t$', fontsize=20, labelpad=-10, loc='top', rotation=0)
ax3.set_xlim(0, xt_vals_new[-1])
ax3.set_ylim(0, kt_vals_plot[-1])
# Add "0" at the origin
ax3.annotate('0', xy=(0, 0), xytext=(-10, -10), textcoords='offset points',
            fontsize=18, ha='right', va='top')
ax3.vlines(x=x_ss_old, ymin=0, ymax=k_ss_old, linestyle=':')
ax3.vlines(x=x_ss_new, ymin=0, ymax=k_ss_new, linestyle=':')

# Set ticks to show the steady state values
ax3.set_xticks([x_ss_old, x_ss_new])
ax3.set_xticklabels([r'$x_1^*$', r'$x_2^*$'], fontsize=14)
ax3.set_yticks([k_ss_old, k_ss_new])
ax3.set_yticklabels([r'$k_1^*$', r'$k_2^*$'], fontsize=14)
# annotate the steady state
ax3.annotate(r'old steady state', xy=(x_ss_old+0.01, k_ss_old+0.05), xytext=(x_ss_old+0.05, k_ss_old+0.3), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))
ax3.annotate(r'new steady state', xy=(x_ss_new+0.009, k_ss_new-0.05), xytext=(x_ss_new+0.05, k_ss_new-0.3), arrowprops=dict(facecolor='black', arrowstyle='->',ls='--'))


# set arrows pointing 
ax3.annotate('', xy=(x_ss_old+0.08, 1.5), xytext=(x_ss_new + 0.05, 1.5), arrowprops=dict(facecolor='black', arrowstyle='<-'))
ax3.annotate('', xy=(x_ss_old+0.1, k_ss_old-0.01), xytext=(x_ss_old+0.1, k_ss_new), arrowprops=dict(facecolor='black', arrowstyle='<-'))
# Add a legend
ax3.legend(loc='upper right', fontsize=12)

# Clean up the plot
ax3.spines[['right', 'top']].set_visible(False)

# Save the figure
plt.tight_layout()
plt.savefig('fig_kPhi_kPsi_sigma_changes_part_updated.pdf', bbox_inches='tight', dpi=300)