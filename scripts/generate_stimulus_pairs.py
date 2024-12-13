import numpy as np
import pandas as pd

import argparse

# testing range constants
SAMPLE_RANGE = 16
DENSE_RANGE_PROPORTION = 0.5
NDENSE_STEPS = 100
NSPARSE_STEPS = 50

# simulation constants
NTRIALS = 100

def compute_range(nsteps, origin, scale):
    step = 1 / (nsteps)
    r = [origin + i * scale * step for i in range(1, nsteps)]
    if scale >= 0:
        return r
    else:
        r.reverse()
        return r
    
def shifted_range(r, shift):
    return [el + shift for el in r]

def get_stest_sample(sref, sample_range = SAMPLE_RANGE, dense_range_proportion = DENSE_RANGE_PROPORTION,
                     ndense_steps = NDENSE_STEPS, nsparse_steps = NSPARSE_STEPS):
    half_range = sample_range // 2
    dense_range_boudary = dense_range_proportion * half_range

    dense_high = compute_range(ndense_steps, sref, dense_range_boudary)
    sparse_high = compute_range(nsparse_steps, sref + dense_range_boudary, (half_range - dense_range_boudary))
    dense_low = compute_range(ndense_steps, sref, - dense_range_boudary)
    sparse_low = compute_range(nsparse_steps, sref - dense_range_boudary, - (half_range - dense_range_boudary))

    return ([sref - half_range] + sparse_low + [sref - dense_range_boudary] + dense_low + [sref]
            + dense_high + [sref + dense_range_boudary] + sparse_high + [sref + half_range])

def generate_stimuli_pairs(sref, stest_stddev, ntrials = NTRIALS):
    columns = ['Trial', 'Stimulus 1 value', 'Stimulus 1 std', 'Stimulus 2 value', 'Stimulus 2 std']

    data = []
    stest_sample = get_stest_sample(sref)
    for t, stest in enumerate(stest_sample):
        for _ in range(ntrials):
            data.append(np.array([t+1, stest, stest_stddev, sref, 0.]))

    df = pd.DataFrame(data=data, columns=columns)
    df['Trial'] = df['Trial'].astype(int)

    return df

def main():
    parser = argparse.ArgumentParser(
                    prog='Stimulus Pairs Generator',
                    description='Generate pairs of stimuli into a csv file')
    parser.add_argument('-sref', '--sref', type=float)
    parser.add_argument('-ststd', '--stest-stddev', type=float)
    opts = parser.parse_args()

    filename = f"stimulus_pairs_sref_{opts.sref}_steststd_{opts.stest_stddev}_ntrials_{NTRIALS}.csv"
    df = generate_stimuli_pairs(opts.sref, opts.stest_stddev)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
