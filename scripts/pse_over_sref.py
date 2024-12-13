import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from generate_graph_bench_results import COLORS, PSE_COLOR

PLOT_ZERO_NOISE = True

# 2000 training
# sref = [-0.3, -0.2, -0.1, 0., 0.1, 0.2]
# pse_0 = [-0.29, -0.20, -0.10, 0.00, 0.10, 0.19]
# pse_05 = [-0.78, -0.49, -0.23, 0.07, 0.34, 0.60]
# pse_1 = [-1.45, -0.88, -0.26, 0.26, 0.83, 1.40]
# pse_15 = [-2.24, -1.28, -0.32, 0.60, 1.53, 2.55]
# pse_2 = [-3.19, -1.79, -0.36, 1.13, 2.57, 3.92]

# more precise
# sref = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
# pse_0 = [-0.31, -0.19, -0.09, 0.01, 0.09, 0.2, 0.31]
# pse_05 = [-0.77, -0.47, -0.21, 0.05, 0.35, 0.62, 0.91]
# pse_1 = [-1.45, -0.83, -0.28, 0.3, 0.85, 1.44, 2.01]
# pse_15 = [-2.2, -1.28, -0.29, 0.71, 1.66, 2.6, 3.56]
# pse_2 = [-3.12, -1.71, -0.21, 1.27, 2.68, 4.14, 5.56]

# no training
sref = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
pse_0 = [-0.3, -0.2, -0.10, -0.01, 0.10, 0.2, 0.3]
pse_05 = [-0.81, -0.56, -0.27, 0.01, 0.27, 0.55, 0.82]
pse_1 = [-1.58, -1.05, -0.51, -0.01, 0.53, 1.06, 1.59]
pse_15 = [-2.46, -1.64, -0.81, 0.0, 0.81, 1.63, 2.45]
pse_2 = [-3.36, -2.28, -1.11, -0.02, 1.13, 2.29, 3.43]

if PLOT_ZERO_NOISE:
    plt.plot(sref, sref, label = "Sref", color=PSE_COLOR, linestyle='--', alpha = 0.5)
    plt.plot(sref, pse_0, label= (r'$\sigma$' + f"(Stest) = {0.}"), color=COLORS[0])
plt.plot(sref, pse_05, label = (r'$\sigma$' + f"(Stest) = {0.5}"), color=COLORS[1])
plt.plot(sref, pse_1, label = (r'$\sigma$' + f"(Stest) = {1.}"), color=COLORS[2])
plt.plot(sref, pse_15, label = (r'$\sigma$' + f"(Stest) = {1.5}"), color=COLORS[3])
plt.plot(sref, pse_2, label = (r'$\sigma$' + f"(Stest) = {2.}"), color=COLORS[4])

f0 = interp1d(sref, pse_0, kind='cubic', fill_value="extrapolate")
f05 = interp1d(sref, pse_05, kind='cubic', fill_value="extrapolate")
f1 = interp1d(sref, pse_1, kind='cubic', fill_value="extrapolate")
f15 = interp1d(sref, pse_15, kind='cubic', fill_value="extrapolate")
f2 = interp1d(sref, pse_2, kind='cubic', fill_value="extrapolate")


initial_guess = (0., 0.)
bbox = dict(boxstyle="round", fc="0.8")
if PLOT_ZERO_NOISE:
    # 0.5 interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq05 = f05(x) - y
        return [eq05, eq0]
    x_prior_05, y_prior_05 = fsolve(equations, initial_guess)
    # plt.plot([x_prior_05, x_prior_05], [min(pse_2), y_prior_05], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    # plt.annotate(f'({x_prior_05:.3f}, {y_prior_05:.3f})', xy=(x_prior_05, y_prior_05), xytext=(x_prior_05 + 0.1, y_prior_05 - 0.5), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)
    # plt.annotate(f'x = {x_prior_05:.3f}', xy=(x_prior_05, y_prior_05), xytext=(x_prior_05 + 0.1, y_prior_05 - 0.5), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 1. interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq1 = f1(x) - y
        return [eq1, eq0]
    x_prior_1, y_prior_1 = fsolve(equations, initial_guess)
    # plt.plot([x_prior_1, x_prior_1], [min(pse_2), y_prior_1], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    # plt.annotate(f'({x_prior_1:.3f}, {y_prior_1:.3f})', xy=(x_prior_1, y_prior_1), xytext=(x_prior_1 + 0.05, y_prior_1 - 1.), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)
    # plt.annotate(f'x = {x_prior_1:.3f}', xy=(x_prior_1, y_prior_1), xytext=(x_prior_1 + 0.05, y_prior_1 - 1.), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 1.5 interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq15 = f15(x) - y
        return [eq15, eq0]
    x_prior_15, y_prior_15 = fsolve(equations, initial_guess)
    # plt.plot([x_prior_15, x_prior_15], [min(pse_2), y_prior_15], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    # plt.annotate(f'({x_prior_15:.3f}, {y_prior_15:.3f})', xy=(x_prior_15, y_prior_15), xytext=(x_prior_15 + -0.1, y_prior_15 + 0.8), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)
    # plt.annotate(f'x = {x_prior_15:.3f}', xy=(x_prior_15, y_prior_15), xytext=(x_prior_15 + -0.1, y_prior_15 + 0.8), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 2. interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq2 = f2(x) - y
        return [eq2, eq0]
    x_prior_2, y_prior_2 = fsolve(equations, initial_guess)
    # plt.plot([x_prior_2, x_prior_2], [min(pse_2), y_prior_2], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    # plt.annotate(f'({x_prior_2:.3f}, {y_prior_2:.3f})', xy=(x_prior_2, y_prior_2), xytext=(x_prior_2 - 0.15, y_prior_2 + 0.3), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)
    # plt.annotate(f'x = {x_prior_2:.3f}', xy=(x_prior_2, y_prior_2), xytext=(x_prior_2 - 0.15, y_prior_2 + 0.3), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

else:
    def equations(vars):
        x, y = vars
        eq05 = f05(x) - y
        eq1 = f1(x) - y
        return [eq05, eq1]

    x_prior_1, y_prior_1 = fsolve(equations, initial_guess)
    plt.plot([x_prior_1, x_prior_1], [min(pse_2), y_prior_1], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_1:.3f}, {y_prior_1:.3f}', xy=(x_prior_1, y_prior_1), xytext=(x_prior_1 + 0.05, y_prior_1 - 0.5), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    def equations(vars):
        x, y = vars
        eq1 = f1(x) - y
        eq15 = f15(x) - y
        return [eq15, eq1]

    x_prior_15, y_prior_15 = fsolve(equations, initial_guess)
    plt.plot([x_prior_15, x_prior_15], [min(pse_2), y_prior_15], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_15:.3f}, {y_prior_15:.3f}', xy=(x_prior_15, y_prior_15), xytext=(x_prior_15 - 0.1, y_prior_15 + 1.), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    def equations(vars):
        x, y = vars
        eq15 = f15(x) - y
        eq2 = f2(x) - y
        return [eq15, eq2]

    x_prior_2, y_prior_2 = fsolve(equations, initial_guess)
    plt.plot([x_prior_2, x_prior_2], [min(pse_2), y_prior_2], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_2:.3f}, {y_prior_2:.3f}', xy=(x_prior_2, y_prior_2), xytext=(x_prior_2 - 0.15, y_prior_2 + 0.2), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)


plt.xlabel("Sref")
plt.ylabel("PSE")
plt.title("Point of subjective equality depending on Sref")
plt.legend()
plt.grid(visible=True)
figname = "pse_over_sref"
if PLOT_ZERO_NOISE:
    figname += "_no_sref"
figname += '.svg'
plt.savefig(figname)
plt.show()