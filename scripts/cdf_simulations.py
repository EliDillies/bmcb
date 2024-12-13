import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from generate_graph_bench_results import COLORS, PSE_COLOR, subjective_point_of_equality

MU_C = -0.1
STD_C = 0.2
PRECISION_C = 1 / (STD_C * STD_C)

STD_OBS = 0.5
PRECISION_OBS = 1. / (STD_OBS * STD_OBS)

MU_CURRECTION = False

def mu_c(std):
    if MU_CURRECTION:
        return MU_C * (1 - std)
    else:
        return MU_C

def precision_tot(std):
    return 1. / (std * std + STD_OBS * STD_OBS)

def mu_estimate_corrected(s, std=0.0):
    precision = precision_tot(std)
    return (precision * s + PRECISION_C * mu_c(std)) / (precision + PRECISION_C)

def mu_estimate(s):
    precision = precision_tot(0.0)
    return (precision * s + PRECISION_C * MU_C) / (precision + PRECISION_C)

def std_estimate(std):
    precision = precision_tot(std)
    return 1. / (precision + PRECISION_C)

def compute_cdf(sref, stest, std):
    std_ref = std_estimate(0.0)
    std_test = std_estimate(std)
    std_tot = math.sqrt(std_test * std_test + std_ref * std_ref)
    point = (mu_estimate_corrected(stest, std) - mu_estimate_corrected(sref)) / std_tot
    return norm.cdf(point)

def cdf_arr(stests, sref, std):
    return [compute_cdf(sref, stest, std) for stest in stests]

def main():
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    stests = np.linspace(-8, 8, 1000)
    stds = [0., 0.5, 1., 1.5, 2.]

    # first plot
    sref = MU_C
    for i, std in enumerate(stds):
        psy = cdf_arr(stests, sref, std)
        axes[0].plot(stests, psy, label = (r'$\sigma$' + f"(Stest) = {std}"), color=COLORS[i])
    
    axes[0].plot([stests.min(), stests.max()], [0.5, 0.5], linestyle='-', color='gray', linewidth=1.5)  # Honrizontal line
    axes[0].annotate(f'y = 1/2', xy=(stests.min(), 0.5), xytext=(stests.min(), 0.55), color=PSE_COLOR)
    axes[0].set_xlabel("Stest")
    axes[0].set_ylabel("P(Stest > Sref)")
    axes[0].set_title(f"Simulated psychometric curves for Sref = " + r'$\mu_c$')
    axes[0].legend()

    # second plot
    srefs = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
    pses = np.zeros((len(stds), len(srefs)))
    for j, sref in enumerate(srefs):
        for i, std in enumerate(stds):
            psy = cdf_arr(stests, sref, std)
            x_pse, _ = subjective_point_of_equality(stests, psy)
            pses[i][j] = x_pse
    
    for i, curve in enumerate(pses):
        axes[1].plot(srefs, curve, label = (r'$\sigma$' + f"(Stest) = {stds[i]}"), color=COLORS[i])
    axes[1].plot(srefs, srefs, label = "Sref", color=PSE_COLOR, linestyle='--', alpha = 0.5)
    axes[1].grid(visible=True)
    axes[1].set_xlabel("Sref")
    axes[1].set_ylabel("PSE")
    axes[1].set_title("Simulated PSE over Sref")

    plt.legend()
    plt.tight_layout()
    plt.savefig('simulations.svg')
    plt.show()

main()