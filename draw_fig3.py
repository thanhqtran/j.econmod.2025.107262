## data about people displaced from jobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

# ==========================================
# ======= Set current folder as root =======
# ==========================================
# Get the directory of the current script
script_dir = pathlib.Path(__file__).resolve().parent
os.chdir(script_dir)

# Parameter values
alpha = 0.33
b = 0.45
h = 0.049
delta = 0.3
z = 0.08
# subparameter
beta = 0.99**30
gamma = 1-beta
eta = (1+gamma+beta)/(1+beta)
xi = h/delta
Q = xi*z*(1+gamma+beta)/gamma
sigma_hat = 1 - np.sqrt(4*Q)
sigma = 0.3

T = 20000
phi_vals = np.linspace(0.4, 1, 3)
x_vals = np.linspace(0.0001, 1-sigma-0.1, T)
k_vals = np.zeros(T)

res = {}

def k_(x, phi):
    theta = (1-x-sigma)**(eta/alpha-1) / ((1-x)**(eta/alpha))
    return phi / ( ( ((1-alpha)/b)**(1/alpha) ) * theta * eta )

def n_(x):
    return h  / (delta * x)

# calculate values
for phi in phi_vals:
    x_values = x_vals
    k_values = k_(x_values, phi)
    n_values = n_(x_values)
    res[phi] = {'x': x_values, 'k': k_values, 'n': n_values}

# Plotting
fig = plt.figure(figsize=(4, 4))
linestyles = ['-', '--', ':']
labels = [r'$\phi_1$', r'$\phi_2$', r'$\phi_3$']
#latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Plotting
ax = plt.subplot(111)
for i in np.arange(len(phi_vals)):
    phi = phi_vals[i]
    ax.plot(res[phi]['x'], res[phi]['k'], label=labels[i], lw=1.5, color='black')

# Hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)
ax.margins(x=0)
ax.margins(y=0)
# add arrows at the end of the axes
# Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
# case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
# respectively) and the other one (1) is an axes coordinate (i.e., at the very
# right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
# actually spills out of the axes.


ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# Add labels and title in latex
ax.set_xlabel(r'$x_t$', fontsize=14, loc='right')
ax.set_ylabel(r'$k_t$', fontsize=14, rotation=0, loc='top')
ax.set_yticks([])
ax.set_xticks([])

# annotate the lines with labels to the right end of the lines
for i in range(len(phi_vals)):
    phi = phi_vals[i]
    ax.annotate(labels[i], (res[phi]['x'][-1], res[phi]['k'][-1]), textcoords="offset points", xytext=(5,0), ha='left', fontsize=14)

# draw the horizontal line from a point to the y-axis
x1 = 0.55
x2 = 0.5
k1 = k_(x1, phi_vals[0])
k2 = k_(x1, phi_vals[1])
k3 = k_(x1, phi_vals[2])
ax.hlines(k2, 0, x1, linestyle='--', color='grey')
ax.vlines(x1, 0, k2, linestyle='--', color='grey')
ax.scatter(x1, k2, color='red', s=20)
ax.scatter(x1, k1, color='red', s=20)
ax.scatter(x2, k_(x2,phi_vals[2]), color='red', s=20)
# annotate the point
ax.annotate(r'$B$', (x1, k2), textcoords="offset points", xytext=(5,0), ha='left')
ax.annotate(r'$C$', (x1, k1), textcoords="offset points", xytext=(5,0), ha='left')
ax.annotate(r'$A$', (x2, k_(x2,phi_vals[2])), textcoords="offset points", xytext=(-8,3), ha='left')

plt.savefig('fig_relation_x_k.pdf', format='pdf', dpi=300)