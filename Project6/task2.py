from scipy.stats import friedmanchisquare, rankdata
import pandas as pd
import numpy as np

algo_perf_df = pd.read_csv("algo_performance.csv")
stat, p = friedmanchisquare(algo_perf_df['C4.5'],
                            algo_perf_df['1-NN'],
                            algo_perf_df['NaiveBayes'],
                            algo_perf_df['Kernel'],
                            algo_perf_df['CN2'])

print('stat=%.3f, p=%s' % (stat, p))

for a in np.arange(0.05, 0, -0.000001):
	if p > a:
		print("For a: {0}. Probably they don't have statically significant differences".format(a))
	else:
		print("For a: {0}. Probably they have statically significant differences".format(a))
