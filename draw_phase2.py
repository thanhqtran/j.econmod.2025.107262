# %%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# use latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
sigma = sigma_hat-0.001
# check if sigma is smaller than sigma_hat
if sigma < sigma_hat:
    print("sigma is smaller than the threshold")

def phi_t(xt, kt):
    return (((1 - alpha) / b) ** (1 / alpha)) * ((1 - xt - sigma) ** ((eta / alpha) - 1)) / ((1 - xt) ** (eta / alpha)) * eta * kt


def xt1_dot(xt, kt):
    return Q/(1-xt-sigma) - xt


def kt1_dot(xt, kt):
    return A * beta * z  * (1 - alpha) * (eta ** alpha) * (kt ** alpha) / (gamma * ((1 - xt - sigma)**alpha)) - kt

def find_kstar(x):
    kk = A*beta*z*(1-alpha)*(eta**alpha)*((1-x-sigma)**(-alpha))/gamma
    return kk**(1/(1-alpha))


def find_xstar():
    Delta = (1-sigma)**2 - 4*Q
    x1 = (1-sigma - np.sqrt(Delta))/2
    x2 = (1-sigma + np.sqrt(Delta))/2
    return x1, x2

# solve for the fixed point of k
def k_star(x):
    sols = fsolve(lambda k: kt1_dot(x, k), 0.6)
    return max(sols)

def x_star(k):
    sols = fsolve(lambda x: xt1_dot(x, k), 0.1)
    return sols



# Create grid
T = 100
xt_vals = np.linspace(0.01, 1 - sigma - 0.1, T)
kt_vals = np.linspace(0.01, 2, T)

kx_vals = np.zeros(T)
xk_vals = np.zeros(T)

# Calculate the nullclines
for i in np.arange(len(xt_vals)):
    x = xt_vals[i]
    kx_vals[i] = k_star(x)

for i in np.arange(len(kt_vals)):
    k = kt_vals[i]
    xk_vals[i] = x_star(k)


X, Y = np.meshgrid(xt_vals, kt_vals)
DX = xt1_dot(X, Y)
DY = kt1_dot(X, Y)
x1, x2 = find_xstar()
k1 = find_kstar(x1)
k2 = find_kstar(x2)
k1star = k_star(x1)

# Plot
plt.figure(figsize=(7, 6))
ax = plt.subplot(111)

# Plot the Nullclines
plt.vlines(x1, 0, kt_vals[-2], color='red', label='x1 nullcline', lw=2)
plt.vlines(x2, 0, kt_vals[-2], color='red', label='x2 nullcline', lw=2)
plt.plot(xt_vals, kx_vals, label='k nullcline', color='green', lw=2)
plt.xlim(0, xt_vals[-1])
plt.ylim(0, kt_vals[-1])


# Plot the phase portrait
plt.streamplot(X, Y, DX, DY, color='b', density=0.5, arrowsize=1.5, arrowstyle='->', minlength=0.2)
plt.scatter(x1, k1, color='red')
plt.scatter(x2, k2, color='orange')
plt.xlabel('$x_t$', fontsize=20, loc='right')
plt.ylabel('$k_t$', fontsize=20, rotation=0, loc='top')

ax.spines[['right', 'top']].set_visible(False)

# Set limits
offset = 0.05
plt.xlim(0, xt_vals[-1]+offset)
plt.ylim(0, kt_vals[-1])

# Axis labels
plt.xlabel(r'$x_t$', fontsize=20, labelpad=-10)
plt.ylabel(r'$k_t$', fontsize=20, rotation=0, labelpad=-10)

# Move axis labels to axis tips
ax.xaxis.set_label_coords(1.02, -0.02)
ax.yaxis.set_label_coords(-0.04, 1.01)

# Arrows at axis tips
ax.annotate('', xy=(xt_vals[-1]+offset, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'), annotation_clip=False)
ax.annotate('', xy=(0, kt_vals[-1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'), annotation_clip=False)

# Label origin with "0"
ax.annotate('0', xy=(0, 0), xytext=(-10, -10), textcoords='offset points',
            fontsize=20, ha='right', va='top')

# annotate on the far right of the plot
plt.annotate('$k_{t+1} = k_t$', (0.38, k_star(0.31)), textcoords="offset points", xytext=(0.38, k_star(0.31)), ha='right', fontsize=20, color='black', rotation=30)

# annotate on top the vertical lines
plt.annotate('$x_{t+1} = x_t$', (x1, kt_vals[-2]), textcoords="offset points",
             xytext=(0, 5), ha='center', fontsize=20, color='black')
plt.annotate('$x_{t+1} = x_t$', (x2, kt_vals[-2]), textcoords="offset points",
             xytext=(0, 5), ha='center', fontsize=20, color='black')

# draw a horizontal line from x1 to the y-axis
plt.hlines(k1, 0, x1, color='grey', ls='--')
plt.vlines(x1, 0, k1, color='grey', ls='--')
plt.hlines(k2, 0, x2, color='grey', ls='--')
plt.vlines(x2, 0, k2, color='grey', ls='--')

plt.xticks([x1, x2], ['$x^*_1$', '$x^*_2$'], fontsize=20)
plt.yticks([k1, k2], ['$k^*_1$', '$k^*_2$'], fontsize=20)

plt.savefig('fig_phase_portrait_full.pdf', bbox_inches='tight', dpi=300)