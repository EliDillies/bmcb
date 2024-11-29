import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from generate_graph_bench_results import COLORS, PSE_COLOR

PLOT_ZERO_NOISE = False

sref = [-0.3, -0.2, -0.1, 0., 0.1, 0.2,]
pse_0 = [-0.28, -0.18, -0.08, 0.02, 0.12, 0.22]
pse_05 = [-0.74, -0.46, -0.22, 0.07, 0.36, 0.64]
pse_1 = [-1.4, -0.88, -0.26, 0.26, 0.87, 1.44]
pse_15 = [-2.2, -1.25, -0.31, 0.63, 1.53, 2.56]
pse_2 = [-3.18, -1.77, -0.36, 1.15, 2.61, 3.92]

if PLOT_ZERO_NOISE:
    plt.plot(sref, sref, label = "Sref", color=PSE_COLOR, linestyle='--', alpha = 0.5)
    plt.plot(sref, pse_0, label= (r'$\sigma$' + f"(Stest) = {0.}"), color=COLORS[0])
plt.plot(sref, pse_05, label = (r'$\sigma$' + f"(Stest) = {0.5}"), color=COLORS[1])
plt.plot(sref, pse_1, label = (r'$\sigma$' + f"(Stest) = {1.}"), color=COLORS[2])
plt.plot(sref, pse_15, label = (r'$\sigma$' + f"(Stest) = {1.5}"), color=COLORS[3])
plt.plot(sref, pse_2, label = (r'$\sigma$' + f"(Stest) = {2.}"), color=COLORS[4])

f0 = interp1d(sref, sref, kind='cubic', fill_value="extrapolate")
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
    plt.plot([x_prior_05, x_prior_05], [min(pse_2), y_prior_05], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_05:.2f}, {y_prior_05:.2f}', xy=(x_prior_05, y_prior_05), xytext=(x_prior_05 + 0.1, y_prior_05 - 0.5), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 1. interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq1 = f1(x) - y
        return [eq1, eq0]
    x_prior_1, y_prior_1 = fsolve(equations, initial_guess)
    plt.plot([x_prior_1, y_prior_1], [min(pse_2), y_prior_1], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_1:.2f}, {y_prior_1:.2f}', xy=(x_prior_1, y_prior_1), xytext=(x_prior_1 + 0.05, y_prior_1 - 1.), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 1.5 interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq15 = f15(x) - y
        return [eq15, eq0]
    x_prior_15, y_prior_15 = fsolve(equations, initial_guess)
    plt.plot([x_prior_15, x_prior_15], [min(pse_2), y_prior_15], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_15:.2f}, {y_prior_15:.2f}', xy=(x_prior_15, y_prior_15), xytext=(x_prior_15 + -0.1, y_prior_15 + 0.8), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # 2. interection with sref:
    def equations(vars):
        x, y = vars
        eq0 = f0(x) - y
        eq2 = f2(x) - y
        return [eq2, eq0]
    x_prior_2, y_prior_2 = fsolve(equations, initial_guess)
    plt.plot([x_prior_2, x_prior_2], [min(pse_2), y_prior_2], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior_2:.2f}, {y_prior_2:.2f}', xy=(x_prior_2, y_prior_2), xytext=(x_prior_2 - 0.15, y_prior_2 + 0.3), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

else:
    def equations(vars):
        x, y = vars
        eq1 = f1(x) - y
        eq15 = f15(x) - y
        return [eq15, eq1]

    x_prior, y_prior = fsolve(equations, initial_guess)
    plt.plot([x_prior, x_prior], [min(pse_2), y_prior], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
    plt.annotate(f'({x_prior:.3f}, {y_prior:.3f}', xy=(x_prior, y_prior), xytext=(x_prior - 0.1, y_prior + 1.), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

plt.xlabel("Sref")
plt.ylabel("PSE")
plt.title("Point of subjective equality depending on Sref")
plt.legend()
figname = "pse_over_sref"
if PLOT_ZERO_NOISE:
    figname += "_no_sref"
plt.savefig(figname)
plt.show()