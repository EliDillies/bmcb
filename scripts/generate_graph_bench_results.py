import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from generate_bench_stimulus_pairs import STDS, NSTDS

COLORS = ['red', 'orange', 'green', 'blue', 'purple']
PSE_COLOR = 'gray'
ANNOTATIONS_COORD = [(0.2, 0.24), (0.6, 0.18), (1., 0.12), (1.4, 0.06), (1.8, 0.)]

def get_parameters_of(filename):
    params = filename.split('_')
    # params = ["results", "sref", "sref_value", "ntrials", "ntrials_value"]
    assert (params[0] == "results")
    return (float(params[2]), int(params[4].split('.')[0]))

def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

def subjective_point_of_equality(x_values, y_values):
    objective_spe = 0.5
    for x, y in zip(x_values, y_values):
        if y >= objective_spe:
            return x, y
    return None, None

def generate_and_save_graph(filename):
    sref, ntrials = get_parameters_of(filename)
    df = pd.read_csv(filename)
    nrows, _ = df.shape
    nsteps = nrows // ntrials // NSTDS

    x = df['Stimulus 1 Value'].unique()
    y = np.empty((nsteps, NSTDS))
    for step in range(nsteps):
        for i in range(NSTDS):
            trials_i = df[df['Trial'] == NSTDS*(step+1)+i]
            y[step, i] = trials_i['Comparison Result'].values.sum() / ntrials

    fine_x = np.linspace(x.min(), x.max(), 300)
    initial_guess = [1, np.median(x), 1]  # Initial parameters: L, midpoint x0, and steepness k

    for i, std in enumerate(STDS):
        y_ = y[:, i]
        params, _ = curve_fit(logistic, x, y_, p0=initial_guess)
        smooth_y = logistic(fine_x, *params)
        x_pse, y_pse = subjective_point_of_equality(fine_x, smooth_y)

        # Data points and interpolated curve
        plt.scatter(x, y_, color=COLORS[i], alpha=0.1)
        plt.plot(fine_x, smooth_y, color=COLORS[i], label=(r'$\sigma$' + f"(Stest) = {std}"))

        # Subjective point of equality
        plt.plot([x_pse, x_pse], [0., y_pse], linestyle='--', color=PSE_COLOR, linewidth=1.5)  # Vertical line
        w, h = ANNOTATIONS_COORD[i]
        bbox = dict(boxstyle="round", fc="0.8")
        plt.annotate(f'x = {x_pse:.2f}', xy=(x_pse, 0.), xytext=(x_pse + w, h), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    plt.plot([x.min(), x.max()], [0.5, 0.5], linestyle='-', color=PSE_COLOR, linewidth=1.5)  # Honrizontal line
    plt.annotate(f'y = 1/2', xy=(x.min(), 0.5), xytext=(x.min(), 0.55), color=PSE_COLOR)
    plt.scatter([sref], [0.], color=PSE_COLOR, marker='+', label=(f'Sref = {sref}')) # Sref
    # plt.annotate(f'Sref = {sref}', xy=(sref, 0.), xytext=(sref-0.6, -0.15), arrowprops=dict(arrowstyle='->', lw=0.8), bbox=bbox)

    # Figure stuffs
    plt.xlabel("Stest")
    plt.ylabel("P(Stest > Sref)")
    plt.title(f"Psychometric functions for Sref = {sref}")
    plt.legend()
    figname = f"psych_fun_sref_{sref}_ntrials_{ntrials}.png"
    plt.savefig(figname)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
                    prog='Stimulus Pairs Results Analyzer',
                    description='Analyze the results of a simulation from a csv file')
    parser.add_argument('-f', '--filename', type=str)
    opts = parser.parse_args()

    generate_and_save_graph(opts.filename)
    print("Graph saved!")

if __name__ == '__main__':
    main()
