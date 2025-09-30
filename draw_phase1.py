# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# parameters
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

# find sigma threshold
def find_sigma_thres(SIGMA):
    X = 0.5 * ((1 - SIGMA) - np.sqrt((1 - SIGMA)**2 - 4 * Q))
    expo = eta / alpha - 1 - alpha / (1 - alpha)
    return 1 - (((1 - alpha) / b)**(1 / alpha) * eta * ((A * beta * z * (1 - alpha) * (eta**alpha) / gamma)**(1 / (1 - alpha))) * ((1 - X - SIGMA)**expo) / ((1 - X)**(eta / alpha)))

def solve_sigma_thres():
    guess = 0.1
    result = fsolve(lambda SIGMA: find_sigma_thres(SIGMA), guess)
    return result[0]

sigma1 = 1 - np.sqrt(4 * Q)
sigma2 = solve_sigma_thres()
sigma_hat = min(sigma1, sigma2)   # corresponds to sigma hat in the paper

# key params
sigma = sigma_hat + 0.1

def phi_t(xt, kt):
    return (((1 - alpha) / b) ** (1 / alpha)) * ((1 - xt - sigma) ** ((eta / alpha) - 1)) / ((1 - xt) ** (eta / alpha)) * eta * kt

# law of motion for x (Psi(k,x) = x)
def xt1_dot(xt, kt):
    return Q / (1 - xt - sigma * phi_t(xt, kt)) - xt

# law of motion for k (Phi(k,x) = k)
def kt1_dot(xt, kt):
    return (A * beta * z * b / gamma) * ((1 - xt) * ((1 - xt - sigma)**(1 - eta)) * phi_t(xt, kt) + (1 - xt) * (1 - phi_t(xt, kt))) / (1 - xt - sigma * phi_t(xt, kt)) - kt

# given x, find k such that Phi(k,x) = k (k-nullcline)
def find_kstar(x):
    # Use a reasonable starting guess for k, e.g., 0.6
    sol, _, ier, _ = fsolve(lambda k: kt1_dot(x, k), 0.6, full_output=True)
    # Check if the solver was successful
    if ier == 1 and sol > 0:
        return sol
    else:
        return np.nan

# given k, find x such that Psi(k,x) = x (x-nullcline)
def find_xstar(k):
    # Use a reasonable starting guess for x, e.g., 0.1
    sol, _, ier, _ = fsolve(lambda x: xt1_dot(x, k), 0.1, full_output=True)
    # Check if the solver was successful
    if ier == 1 and sol > 0:
        return sol
    else:
        return np.nan

def find_steady_state(vars):
    """
    Function to find the root of the system of equations.
    The steady state is where both laws of motion are zero.
    """
    x, k = vars
    return [xt1_dot(x, k), kt1_dot(x, k)]

# --- Calculation and Plotting ---

# Find the steady-state equilibrium point by solving the system
initial_guess = [0.1, 0.5] # Start solver with a reasonable guess
steady_state_solution = fsolve(find_steady_state, initial_guess)
x11, k11 = steady_state_solution
print(f"Calculated Steady State: x* = {x11:.4f}, k* = {k11:.4f}")

# Create grid
T = 1000
xt_vals = np.linspace(0.01, 1 - sigma - 0.01, T)
kt_vals = np.linspace(0.01, 2, T)

# Use LaTeX for plot labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# METHOD 1: Calculate the nullclines (NUMERICAL APPROACH)
kx_vals = np.zeros(T)
xk_vals = np.zeros(T)

# the nullcline for k
for i, x in enumerate(xt_vals):
    kx_vals[i] = find_kstar(x)

# the nullcline for x
for i, k in enumerate(kt_vals):
    xk_vals[i] = find_xstar(k)

# Create meshgrid for the streamplot
X, Y = np.meshgrid(xt_vals, kt_vals)
DX = xt1_dot(X, Y)
DY = kt1_dot(X, Y)

# Plot the phase portrait
plt.figure(figsize=(7, 6))
ax = plt.subplot(111)

# Plot the vector field
plt.streamplot(X, Y, DX, DY, color='b', density=0.7, arrowsize=1.2, linewidth=0.8)

# Plot the nullclines
plt.plot(xt_vals, kx_vals, label=r'$k_{t+1}=k_t$ (k-nullcline)', color='green', lw=2.5)
plt.plot(xk_vals, kt_vals, label=r'$x_{t+1}=x_t$ (x-nullcline)', color='red', lw=2.5)

# Set plot labels and limits
plt.xlabel(r'$x_t$', fontsize=15)
plt.ylabel(r'$k_t$', fontsize=15, rotation=0, labelpad=15)
plt.xlim(0, xt_vals[-1])
plt.ylim(0, kt_vals[-1])

# --- FIXES APPLIED BELOW ---

# Plot the steady state point and guidelines
plt.scatter(x11, k11, color='black', s=50, zorder=5)
plt.hlines(k11, 0, x11, color='grey', ls='--')
plt.vlines(x11, 0, k11, color='grey', ls='--')

# Annotate the nullclines
# Corrected the function name from k_star to find_kstar and added [0]
# Positioned the annotation for clarity
# annotate on the far right of the plot
plt.annotate('$k_{t+1} = k_t$', (0.3, k11+0.05), textcoords="offset points", 
        xytext=(0.3, k11+0.05), ha='right', fontsize=20, color='black', rotation=0)

# annotate on top the vertical lines
plt.annotate('$x_{t+1} = x_t$', (x11+0.05, kt_vals[-2]), textcoords="offset points",
             xytext=(0, 5), ha='center', fontsize=20, color='black')


# Set the ticks to show the steady state values
plt.xticks([x11], [r'$x^*$'], fontsize=14)
plt.yticks([k11], [r'$k^*$'], fontsize=14)

# Remove top and right spines for a cleaner look
ax.spines[['right', 'top']].set_visible(False)

# Remove default axis labels
plt.xlabel(r'$x_t$', fontsize=20, labelpad=-10, loc='right')
plt.ylabel(r'$k_t$', fontsize=20, labelpad=-10, loc='top')

# Set axis limits
plt.xlim(0, xt_vals[-1])
plt.ylim(0, kt_vals[-1])

# Add arrows at the end of axes
ax.annotate('', xy=(xt_vals[-1], 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='black'), annotation_clip=False)
ax.annotate('', xy=(0, kt_vals[-1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='black'), annotation_clip=False)

# Axis labels at the tips
# ax.annotate(r'$x_t$', xy=(xt_vals[-1], 0), xytext=(0, -20), textcoords='offset points',
#             ha='right', va='top', fontsize=20)
# ax.annotate(r'$k_t$', xy=(0, kt_vals[-1]), xytext=(10, 0), textcoords='offset points',
#             ha='left', va='bottom', fontsize=20)
# Add "0" at the origin
ax.annotate('0', xy=(0, 0), xytext=(-10, -10), textcoords='offset points',
            fontsize=18, ha='right', va='top')


# Update tick label sizes
plt.xticks([x11], [r'$x^*$'], fontsize=20)
plt.yticks([k11], [r'$k^*$'], fontsize=20)


# Scatter dot and guidelines for steady state
plt.scatter(x11, k11, color='black', s=60, zorder=5)
plt.hlines(k11, 0, x11, color='grey', ls='--')
plt.vlines(x11, 0, k11, color='grey', ls='--')

# Clean layout and remove top/right spines
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()

plt.savefig('fig_phase_portrait_part.pdf', bbox_inches='tight', dpi=300)