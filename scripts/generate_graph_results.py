import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def get_parameters_of(filename):
    params = filename.split('_')
    # params = ["results", "sref", "sref_value", "ststd", "stest_stddev", "ntrials", "ntrials_value"]
    assert (params[0] == "results")
    return (float(params[2]), float(params[4]), int(params[6].split('.')[0]))

def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def subjective_point_of_equality(x_values, y_values):
    objective_spe = 0.5
    for x, y in zip(x_values, y_values):
        if y >= objective_spe:
            return x, y
    return None, None

def generate_and_save_graph(filename):
    sref, ststd, ntrials = get_parameters_of(filename)
    df = pd.read_csv(filename)
    nrows, _ = df.shape
    nsteps = nrows // ntrials

    x, y = np.empty(nsteps), np.empty(nsteps)
    for st in range(nsteps):
        trials_i = df[df['Trial'] == st+1]
        x[st] = trials_i['Stimulus 1 Value'].iloc[0]
        y[st] = trials_i['Comparison Result'].values.sum() / ntrials

    fine_x = np.linspace(x.min(), x.max(), 300)
    initial_guess = [1, np.median(x), 1]  # Initial parameters: L, midpoint x0, and steepness k
    params, _ = curve_fit(logistic, x, y, p0=initial_guess)
    smooth_y = logistic(fine_x, *params)
    x_pse, y_pse = subjective_point_of_equality(fine_x, smooth_y)

    # Data points and interpolated curve
    plt.scatter(x, y, color='red')
    plt.plot(fine_x, smooth_y)

    # Subjective point of equality
    plt.plot([x_pse, x_pse], [0., y_pse], linestyle='--', color='gray', linewidth=1.5)  # Vertical line
    plt.plot([x.min(), x.max()], [0.5, 0.5], linestyle='-', color='orange', linewidth=1.5)  # Horizontal line
    plt.annotate(f'y = 1/2', xy=(x.min(), 0.5), xytext=(x.min(), 0.55), color='orange')
    plt.annotate(f'x = {x_pse:.2f}', xy=(x_pse, 0.), xytext=(x_pse + 0.5, 0.1), arrowprops=dict(arrowstyle='->', lw=0.8))
    plt.scatter([sref], [0.], color='orange', marker='+')
    plt.annotate(f'Sref = {sref}', xy=(sref, 0.), xytext=(sref-0.6, -0.2), arrowprops=dict(arrowstyle='->', lw=0.8))

    # Figure stuffs
    plt.title(f"Psychometric function for Sref = {sref} and " + r'$\sigma$' + f"(Stest) = {ststd}")
    figname = f"psych_fun_sref_{sref}_steststd_{ststd}_ntrials_{ntrials}.png"
    plt.savefig(figname)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
                    prog='Stimulus Pairs Results Analyzer',
                    description='Analyze the results of a simulation from a csv file')
    parser.add_argument('-f', '--filename', type=str)
    opts = parser.parse_args()

    generate_and_save_graph(opts.filename)

if __name__ == '__main__':
    main()
